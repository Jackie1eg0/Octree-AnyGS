#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
render.py — Octree-GS / Scaffold-GS 渲染管线

本文件定义了从 3D高斯场景 到 2D图像 的完整渲染管线。
包含 3DGS 和 2DGS 两种渲染模式, 每种模式由两个函数组成:

┌─────────────────────────────────────────────────────────────────────────┐
│                          渲染管线流程 (每帧)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step1: set_anchor_mask()   → LOD层级筛选 (lod_model.py)                 │
│         计算相机到每个Anchor的距离, 筛选 anchor_level ≤ int_level         │
│                          ↓                                              │
│  Step2: prefilter_voxel()   → Anchor级视锥剔除 (本文件)                   │
│         将Anchor当作粗粒度3D椭球投影到屏幕, radii>0的Anchor保留            │
│         结果: visible_mask [N] bool,同时编码了LOD筛选+视锥剔除            │
│                          ↓                                              │
│  Step3: generate_neural_gaussians() → MLP解码 (lod_model.py)             │
│         对可见Anchor用MLP预测颜色/不透明度/协方差,生成Neural Gaussians      │
│         结果: xyz, color, opacity, scaling, rot, selection_mask          │
│                          ↓                                              │
│  Step4: gsplat.rasterization() → CUDA光栅化 (gsplat库)                   │
│         3D Gaussian Splatting到2D屏幕, alpha-blending逐像素着色            │
│         结果: rendered_image, radii, means2d                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

函数列表:
  - render()              : 3DGS渲染主函数 (调用prefilter_voxel)
  - prefilter_voxel()     : 3DGS的Anchor视锥剔除
  - render_2dgs()         : 2DGS渲染主函数 (调用prefilter_voxel_2dgs)
  - prefilter_voxel_2dgs(): 2DGS的Anchor视锥剔除
"""

import torch
import math
import gsplat
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs


# =============================================================================
# 3DGS 渲染主函数
# =============================================================================
def render(viewpoint_camera, pc, pipe, bg_color, iteration, render_mode, ape_code=-1):
    """3D Gaussian Splatting 渲染主函数。每帧训练/推理时调用。
    
    完整流程:
      1. set_anchor_mask()          → LOD筛选: 计算相机-Anchor距离, 确定哪些Anchor在该视角的LOD层级内
      2. prefilter_voxel()          → 视锥剔除: 将Anchor当作椭球投影到屏幕, 剔除radii=0的Anchor点
      3. generate_neural_gaussians() → MLP解码: 对可见Anchor的特征输入MLP, 解码出Neural Gaussian属性
      4. gsplat.rasterization()      → 光栅化: 3D Gaussian Splatting到屏幕, alpha-blending着色
    
    参数:
        viewpoint_camera: Camera对象, 包含:
            - camera_center [3]:     相机世界坐标
            - world_view_transform:  W2C变换矩阵 [4,4]
            - FoVx, FoVy:            水平/垂直视场角(弧度)
            - image_width, image_height: 渲染目标分辨率
            - resolution_scale:      分辨率缩放因子
        pc:         GaussianLoDModel(或GaussianModel), 高斯模型对象
        pipe:       PipelineParams, 渲染管线参数
        bg_color:   [3] GPU张量, 背景颜色 (通常为黑色[0,0,0]或白色[1,1,1])
        iteration:  int, 当前训练迭代步 (用于渐进训练的LOD层级限制)
        render_mode: str, 渲染模式:
            - "RGB":    只渲染颜色
            - "RGB+ED": 渲染颜色+深度 (Expected Depth)
        ape_code:   int, 外观嵌入的相机索引 (默认-1表示不使用外观嵌入)
    
    返回:
        dict, 包含:
            render:            [3, H, W] 渲染的RGB图像
            scaling:           [M, 3] 每个Neural Gaussian的缩放参数
            viewspace_points:  [1, M, 2] 每个Gaussian的屏幕空间2D坐标(means2d), 带梯度
                               用于 training_statis() 计算位置梯度, 指导anchor_growing
            visibility_filter: [M] bool, 光栅化时radii>0的Gaussian (真正参与了像素着色的)
            visible_mask:      [N] bool, 可见Anchor掩码 (LOD筛选 + 视锥剔除)
            selection_mask:    [N*k] bool, MLP预测Opacity>0的有效Gaussian掩码
            opacity:           [M, 1] 每个Gaussian的不透明度
            render_depth:      [1, H, W] 深度图 (仅RGB+ED模式), 或 None
    """
    # =========================================================================
    # Step 1: LOD层级筛选 — 根据相机到Anchor的距离确定LOD可见层级
    # =========================================================================
    # 计算每个Anchor的预测LOD层级 L* = log2(d_max/dist)/log2(fork) + ΔL
    # 筛选条件: anchor_level ≤ int_level, 结果存入 pc._anchor_mask [N] bool
    pc.set_anchor_mask(viewpoint_camera.camera_center, iteration, viewpoint_camera.resolution_scale)
    
    # =========================================================================
    # Step 2: Anchor级视锥剔除 — 将Anchor当作粗粒度椭球投影,看是否在屏幕内
    # =========================================================================
    # 输入: pc._anchor_mask (LOD筛选后的Anchor)
    # 输出: visible_mask [N] bool, 同时编码了LOD筛选+视锥剔除
    #        只有 _anchor_mask=True 且 radii>0 的Anchor才为True
    visible_mask = prefilter_voxel(viewpoint_camera, pc, pipe, bg_color).squeeze()
    
    # =========================================================================
    # Step 3: MLP解码 — 对可见Anchor生成Neural Gaussians
    # =========================================================================
    # 输入: visible_mask (可见Anchor), ape_code (外观嵌入索引)
    # 过程: 提取可见Anchor的特征 → 计算相机到Anchor方向 → MLP解码颜色/opacity/协方差
    #       selection_mask 标记了Opacity>0的有效Gaussian (MLP可以动态关闭某些GS)
    # 输出: xyz[M,3], color[M,3], opacity[M,1], scaling[M,3], rot[M,4]
    #        M = selection_mask中True的数量 (有效Neural Gaussian数)
    xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_neural_gaussians(viewpoint_camera, visible_mask, ape_code)
    
    # =========================================================================
    # Step 4: CUDA光栅化 — 3D Gaussian Splatting + Alpha-Blending
    # =========================================================================
    # 构建相机内参矩阵 K [3,3]
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)   # fx = W / (2·tan(FoVx/2))
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)   # fy = H / (2·tan(FoVy/2))
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],   # [fx,  0, cx]
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],  # [ 0, fy, cy]
            [0, 0, 1],                                                  # [ 0,  0,  1]
        ],
        device="cuda",
    )
    
    # W2C视图矩阵 [4,4], 注意gsplat要求行主序所以做转置
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    
    # 调用gsplat库的CUDA光栅化器
    # 内部流程: 
    #   1) 3D Gaussian → 2D Gaussian (Splatting投影)
    #   2) 计算每个2D Gaussian的radii (屏幕覆盖半径)
    #   3) Tile-Based Rasterization: 分块 → 深度排序 → 逐像素alpha-blending
    render_colors, render_alphas, info = gsplat.rasterization(
        means=xyz,                              # [M, 3] Gaussian世界坐标
        quats=rot,                              # [M, 4] 旋转四元数
        scales=scaling,                         # [M, 3] 缩放参数
        opacities=opacity.squeeze(-1),          # [M,]   不透明度
        colors=color,                           # [M, 3] 或 [M, SH_coeffs] 颜色
        viewmats=viewmat[None],                 # [1, 4, 4] W2C矩阵 (batch=1)
        Ks=K[None],                             # [1, 3, 3] 内参矩阵 (batch=1)
        backgrounds=bg_color[None],             # [1, 3] 背景颜色
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,                           # 使用非打包模式 (更快但占更多显存)
        sh_degree=sh_degree,                    # SH阶数 (Scaffold-GS中通常为0,用MLP替代SH)
        render_mode=render_mode,                # "RGB" 或 "RGB+ED"
    )

    # =========================================================================
    # Step 5: 解析光栅化结果
    # =========================================================================
    # render_colors: [1, H, W, 3或4], 最后一维为3时只有RGB, 为4时包含深度
    if render_colors.shape[-1] == 4:
        # RGB+ED模式: 前3通道=颜色, 第4通道=期望深度(Expected Depth)
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)      # [1, H, W, 1] → [1, H, W]
    else:
        # RGB模式: 无深度输出
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)  # [1, H, W, 3] → [3, H, W]
    radii = info["radii"].squeeze(0)              # [1, M] → [M,] 每个Gaussian的投影半径
    
    # 保留 means2d 的梯度 — 这是致密化统计的关键
    # means2d 是3D坐标经Splatting投影到屏幕的2D坐标, 不是叶子节点, 需要手动retain_grad
    # Loss反传时, ∂Loss/∂means2d 表示"渲染误差推动该Gaussian移动的力"
    # training_statis() 会读取这个梯度来判断哪些区域需要新增Anchor
    try:
        info["means2d"].retain_grad() # [1, M, 2]
    except:
        pass

    # 构建返回字典 — training_statis() 和 train.py 会用到其中的各项
    return_dict = {
        "render": rendered_image,                # [3, H, W] 渲染的RGB图像, 与gt图像计算Loss
        "scaling": scaling,                      # [M, 3] Neural Gaussian缩放 (仅传递, 不直接用于统计)
        "viewspace_points": info["means2d"],     # [1, M, 2] 2D屏幕坐标, .grad用于致密化梯度统计
        "visibility_filter" : radii > 0,         # [M] bool, radii>0 = 真正参与像素着色的GS (Step4筛选)
        "visible_mask": visible_mask,            # [N] bool, 可见Anchor (Step1 LOD + Step2 视锥)
        "selection_mask": selection_mask,         # [N*k] bool, MLP预测有效的GS (Step3, Opacity>0)
        "opacity": opacity,                      # [M, 1] 不透明度, 用于opacity_accum统计 → 剪枝决策
        "render_depth": depth                    # [1, H, W] 深度图 (RGB+ED模式) 或 None
    }
    
    return return_dict


# =============================================================================
# 3DGS Anchor级视锥剔除
# =============================================================================
def prefilter_voxel(viewpoint_camera, pc, pipe, bg_color):
    """Anchor级视锥剔除 (3DGS版)。
    
    将每个Anchor当作一个粗粒度的3D椭球体(由Anchor的scaling前3维和rotation定义),
    通过gsplat的投影函数将其Splatting到2D屏幕。如果投影后radii>0(在屏幕上有覆盖),
    则认为该Anchor"可见"; radii=0 则被剔除。
    
    注意: 输入已经过 _anchor_mask 筛选(LOD层级过滤), 本函数在此基础上进一步做视锥剔除。
    输出的 visible_mask 同时编码了两层筛选的结果。
    
    为什么不直接对所有Gaussian做视锥剔除?
      因为每个Anchor管辖k个Gaussian, 如果Anchor本身不在视锥内, 其管辖的GS大概率也不在,
      先剔除Anchor可以大幅减少后续MLP推理的计算量 (避免对不可见Anchor做MLP解码)。
    
    参数:
        viewpoint_camera: Camera对象, 包含相机内外参和图像尺寸
        pc:       GaussianLoDModel, 高斯模型对象
        pipe:     PipelineParams (本函数未使用, 保留接口一致性)
        bg_color: 背景颜色 (本函数未使用, 保留接口一致性)
    
    返回:
        visible_mask: [N] bool, True=该Anchor通过了LOD筛选+视锥剔除, 其管辖的GS可以进入MLP解码
    """
    # 提取经过LOD筛选(_anchor_mask=True)的Anchor属性
    means = pc.get_anchor[pc._anchor_mask]              # [N', 3] LOD可见Anchor的世界坐标
    scales = pc.get_scaling[pc._anchor_mask][:, :3]     # [N', 3] Anchor缩放的前3维(定义椭球范围)
    quats = pc.get_rotation[pc._anchor_mask]            # [N', 4] Anchor旋转四元数
    
    # 构建相机内参矩阵 K [1, 3, 3]
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    Ks = torch.tensor([
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],device="cuda",)[None]                         # [1, 3, 3]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]  # [1, 4, 4] W2C矩阵

    N = means.shape[0]      # LOD可见的Anchor数量
    C = viewmats.shape[0]   # 相机数量 (这里固定为1)
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # 将Anchor(椭球)投影到2D屏幕 — 这不是完整的光栅化,只计算投影参数
    # 核心输出是 radii: 每个Anchor在屏幕上的投影半径
    #   radii > 0 → 该Anchor的椭球投影到屏幕上有覆盖面积 → 可见
    #   radii = 0 → 不在屏幕内/深度在近远平面外/投影面积为0 → 不可见
    proj_results = fully_fused_projection(
        means,
        None,       # covars: 不预计算协方差,直接用quats+scales更快
        quats,
        scales,
        viewmats,
        Ks,
        int(viewpoint_camera.image_width),
        int(viewpoint_camera.image_height),
        eps2d=0.3,          # 2D协方差的最小特征值下限,防止退化
        packed=False,
        near_plane=0.01,    # 近平面距离
        far_plane=1e10,     # 远平面距离
        radius_clip=0.0,    # radii剪裁阈值
        sparse_grad=False,
        calc_compensations=False,
    )
    
    # 解析投影结果: radii [C, N] = [1, N']
    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None
    
    # 构建最终 visible_mask [N] — 在 _anchor_mask 基础上叠加视锥剔除结果
    # 初始化: 复制 _anchor_mask (LOD筛选结果)
    # 然后在 _anchor_mask=True 的位置,用 radii>0 进一步过滤
    visible_mask = pc._anchor_mask.clone()                      # [N] bool, 从LOD筛选结果开始
    visible_mask[pc._anchor_mask] = radii.squeeze(0) > 0       # LOD通过的Anchor中, radii>0的才保留
    
    return visible_mask


# =============================================================================
# 2DGS 渲染主函数
# =============================================================================
def render_2dgs(viewpoint_camera, pc, pipe, bg_color, iteration, render_mode):
    """2D Gaussian Splatting 渲染主函数。流程与 render() 基本一致。
    
    与3DGS的主要区别:
      1. 使用 gsplat.rasterization_2dgs() 替代 gsplat.rasterization()
      2. 额外返回法线(normals)、深度法线(normals_from_depth)、变形损失(distort)
      3. 强制使用 "RGB+ED" 渲染模式
      4. 不支持外观嵌入 (无ape_code参数)
    
    参数:
        viewpoint_camera: Camera对象
        pc:         GaussianLoDModel, 高斯模型
        pipe:       PipelineParams, 管线参数
        bg_color:   [3] 背景颜色
        iteration:  int, 当前训练步
        render_mode: str, 必须为 "RGB+ED"
    
    返回:
        dict, 包含:
            render:                    [3, H, W]  渲染RGB图像
            scaling:                   [M, 3]     Neural Gaussian缩放
            viewspace_points:          [1, M, 2]  2D屏幕坐标 (带梯度)
            visibility_filter:         [M] bool   radii>0的Gaussian
            visible_mask:              [N] bool   可见Anchor (LOD+视锥)
            selection_mask:            [N*k] bool  有效Gaussian (Opacity>0)
            opacity:                   [M, 1]     不透明度
            render_depth:              [1, H, W]  深度图
            render_normals:            [1, H, W, 3] 渲染法线
            render_alphas:             [1, H, W, 1] 渲染透明度
            render_normals_from_depth: [1, H, W, 3] 从深度图推算的法线
            render_distort:            [1, H, W, 1] 变形损失 (2DGS正则化项)
    """
    assert render_mode=="RGB+ED", "Only RGB+ED mode is supported for 2D Gaussians."
    
    # Step 1-3: 与3DGS相同 (LOD筛选 → 视锥剔除 → MLP解码)
    pc.set_anchor_mask(viewpoint_camera.camera_center, iteration, viewpoint_camera.resolution_scale)
    visible_mask = prefilter_voxel_2dgs(viewpoint_camera, pc, pipe, bg_color).squeeze()
    xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_neural_gaussians(viewpoint_camera, visible_mask)
    
    # 构建相机内参矩阵
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )
    
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    
    # Step 4: 2DGS光栅化 — 额外输出法线和变形损失
    (render_colors, 
    render_alphas,
    render_normals,                 # [1, H, W, 3] 从2D Gaussian法线混合得到的法线图
    render_normals_from_depth,      # [1, H, W, 3] 由深度图通过求梯度推算的法线
    render_distort,                 # [1, H, W, 1] 变形损失: 惩罚沿射线方向的Gaussian分散程度
    render_median,), info = \
    gsplat.rasterization_2dgs(
        means=xyz,                              # [M, 3]
        quats=rot,                              # [M, 4]
        scales=scaling,                         # [M, 3] 注意: 2DGS中第3维通常被压缩为极小值
        opacities=opacity.squeeze(-1),          # [M,]
        colors=color,
        viewmats=viewmat[None],                 # [1, 4, 4]
        Ks=K[None],                             # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )

    # 解析光栅化结果 (与3DGS相同)
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)     # [3, H, W]
    radii = info["radii"].squeeze(0)                 # [M,]
    try:
        info["means2d"].retain_grad()                # [1, M, 2]
    except:
        pass

    return_dict = {
        "render": rendered_image,                    # [3, H, W] RGB图像
        "scaling": scaling,                          # [M, 3]
        "viewspace_points": info["means2d"],         # [1, M, 2] 带梯度的2D坐标
        "visibility_filter" : radii > 0,             # [M] bool
        "visible_mask": visible_mask,                # [N] bool
        "selection_mask": selection_mask,             # [N*k] bool
        "opacity": opacity,                          # [M, 1]
        "render_depth": depth,                       # [1, H, W]
        "render_normals": render_normals,            # [1, H, W, 3] 渲染法线
        "render_alphas": render_alphas,              # [1, H, W, 1] 累积透明度
        "render_normals_from_depth": render_normals_from_depth,  # [1, H, W, 3]
        "render_distort": render_distort,            # [1, H, W, 1] 变形正则化损失
    }
    
    return return_dict


# =============================================================================
# 2DGS Anchor级视锥剔除
# =============================================================================
def prefilter_voxel_2dgs(viewpoint_camera, pc, pipe, bg_color):
    """Anchor级视锥剔除 (2DGS版)。
    
    功能与 prefilter_voxel() 完全一致: 将Anchor投影到屏幕,
    radii>0的保留, radii=0的剔除。
    
    与3DGS版的唯一区别: 使用 fully_fused_projection_2dgs() 替代 fully_fused_projection(),
    2DGS的投影需要额外传入 densifications 参数 (用于密度补偿)。
    
    参数:
        viewpoint_camera: Camera对象
        pc:       GaussianLoDModel, 高斯模型
        pipe:     PipelineParams (未使用)
        bg_color: 背景颜色 (未使用)
    
    返回:
        visible_mask: [N] bool, 通过LOD+视锥双重筛选的Anchor掩码
    """
    # 提取LOD可见Anchor的属性
    means = pc.get_anchor[pc._anchor_mask]
    scales = pc.get_scaling[pc._anchor_mask][:, :3]
    quats = pc.get_rotation[pc._anchor_mask]
    
    # 构建相机参数
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    Ks = torch.tensor([
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],device="cuda",)[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # 2DGS特有: 密度补偿参数 (prefilter阶段不需要实际值, 用零填充)
    densifications = (
        torch.zeros((C, N, 2), dtype=means.dtype, device="cuda")
    )
    
    # 2DGS投影: 与3DGS类似,但使用2DGS专用的投影函数
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        densifications,
        Ks,
        int(viewpoint_camera.image_width),
        int(viewpoint_camera.image_height),
        eps2d=0.3,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        sparse_grad=False,
    )
    
    # 构建 visible_mask (逻辑与3DGS版完全一致)
    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii.squeeze(0) > 0
    
    return visible_mask
