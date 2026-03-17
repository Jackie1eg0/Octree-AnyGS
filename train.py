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
# ============================================================================
# Octree-AnyGS 训练主入口
# 本文件是整个 Octree-GS 系统的训练/渲染/评估入口。
# 核心流程: 初始化场景 → 渐进式训练循环 → 渲染 → 评估指标
# 
# 【Octree-GS 关键步骤在本文件中的体现】:
#   阶段一(Octree构建与Anchor初始化):  Scene() 会调用 gaussians.create_from_pcd() 构建八叉树
#   阶段三(渐进训练:从粗到精（Coarse-to-fine）地分配几何细节): gaussians.set_coarse_interval() 设定层级解锁时间表
#   阶段四(自适应控制Grow & Prune: Next-level Growing & View-Frequency Pruning):   训练循环中 training_statis() + run_densify() 
#   阶段二/五(前向渲染与自适应LOD选择/外观特征嵌入): 在 render() 和 generate_neural_gaussians() 中(非本文件)
# ============================================================================

import os
import shutil
import numpy as np

# ---------- 自动选择显存占用最少的 GPU ----------
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import torchvision
import json
import wandb
import time
from datetime import datetime
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim          # L1损失 和 SSIM结构相似性损失
import sys
from gaussian_renderer import network_gui            # GUI远程查看器(Octree-GS中尚不可用)
from scene import Scene                              # 场景管理类Scene, 内部触发八叉树初始化
from utils.general_utils import safe_state, parse_cfg, get_render_func  # 配置解析 + 渲染函数路由
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import yaml
import warnings
warnings.filterwarnings('ignore')

# LPIPS感知损失网络(用于评估阶段计算感知质量指标)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

# TensorBoard 日志支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    """将当前项目代码备份到输出目录, 方便实验复现和代码追溯。
    会读取 .gitignore 排除不需要的文件(如数据集、输出、缓存等)。
    """
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    assert os.path.exists(os.path.join(ROOT, '.gitignore'))
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


# =============================================================================
# training() — 核心训练函数
# =============================================================================
def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    """
    Octree-GS 核心训练循环。
    model/opt/pipe 来源于yaml文件(如config/scaffoldgs/base_model.yaml 或者是 config/scaffoldgs/lod_model.yaml)
    
    参数:
        dataset:    模型参数(model_params), 包含 base_model, model_config 等
        opt:        优化参数(optim_params), 包含 iterations, 致密化阈值等
        pipe:       Pipeline参数
        ply_path:   Warmup模式下, 上一轮训练产出的点云路径
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)  # 初始化TensorBoard

    # =========================================================================
    # 【Octree-GS 阶段一: 模型构建与八叉树初始化】
    # 1. 根据配置文件中的 base_model (scaffoldgs/3dgs/2dgs) 动态导入对应模型
    # 2. 根据 model_config.name (GaussianLoDModel/GaussianModel) 实例化
    #    - GaussianLoDModel: 带LOD的八叉树模型 (Octree-GS核心)
    #    - GaussianModel:    无LOD的基础模型 (Scaffold-GS基线)
    # 
    # Scene/目录下有3个模型文件夹,相关的模型文件在Scene/gs_model_xxx.py中(如base_model.py/lod_model.py)含有GaussianModel类与GaussianLoDModel类定义
    # =========================================================================
    modules = __import__('scene.gs_model_'+dataset.base_model, fromlist=[''])
    model_config = dataset.model_config                                           # YAML文件中的Model_config字典
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])  # getattr()函数用于获取对象属性, 这里获取model_config['name']对应的属性(字典)
                                                                                  # 并将model_config['kwargs'](字典)作为参数传入用于初始化Model
                                                                                  # Eg:假设 base_model="scaffoldgs", model_config['name']="GaussianLoDModel" --> gaussians = GaussianLoDModel(feat_dim=32, n_offsets=10, fork=2, ...) 
    # 【阶段一续】Scene.__init__() 内部会调用:
    #   gaussians.create_from_pcd(pcd, ...)
    #     → set_level()      计算 d_max, d_min, K(LOD层数)
    #     → octree_sample()  对点云进行多层体素化, 构建八叉树
    #     → weed_out()       按可见频率裁剪离群锚点
    #     → 初始化 _anchor, _level, _extra_level 等可学习参数
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales)
    
    # 【Octree-GS 阶段三: 渐进式训练调度 — 设定层级解锁时间表】
    # set_coarse_interval() 根据 coarse_iter(总渐进步数, 默认10000) 和
    # coarse_factor(生长因子ω, 默认1.5) 计算每一层的解锁迭代步数。
    # 训练从 init_level(≈K/2) 层开始, 逐步解锁更精细的层。
    # 粗糙层分配更多训练步数(等比递减), 确保全局结构先收敛。
    gaussians.set_coarse_interval(opt)
    
    # 配置优化器(Adam) 和 各参数组的学习率调度器
    gaussians.training_setup(opt)
    
    # 如果有checkpoint, 恢复训练状态
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # CUDA事件用于精确计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None          # 相机队列, 用完后重新打乱
    ema_loss_for_log = 0.0          # 指数移动平均损失(仅用于显示)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    modules = __import__('gaussian_renderer')  # 导入渲染模块(render/render_2dgs)
    
    # ==========================================================================
    # 主训练循环
    # ==========================================================================
    for iteration in range(first_iter, opt.iterations + 1):        
        # ---------- GUI远程查看器 (Octree-GS中暂不可用) ----------
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.compute_cov3D_python, keep_alive = network_gui.receive()
                if custom_cam != None:
                    net_image = getattr(modules, get_render_func(dataset.base_model))(custom_cam, gaussians, pipe, scene.background, iteration, dataset.render_mode)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # 更新所有参数组的学习率(指数衰减),调用basic_model.py中方法,按照指数衰减策略更新各参数组(Anchor位置、特征、offset等)学习率
        gaussians.update_learning_rate(iteration)
        
        # ---------- 随机采样一个训练相机,从train_cameras中随机不放回采样一个相机 ----------
        # 每个epoch内不重复采样,队列用完后从scene.getTrainCameras()重新获取并打乱 
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # ---------- 前向渲染 ----------------------
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # 【Octree-GS 阶段二: 自适应LOD选择 + 渲染】
        # get_render_func() 根据 base_model 返回 "render" 或 "render_2dgs"
        # 渲染内部流程 (gaussian_renderer/render.py):
        #   1. set_anchor_mask()  → 根据相机距离计算每个锚点的连续LOD等级
        #                           L* = log2(d_max/d) + ΔL, 筛选 level ≤ floor(L*)
        #                           【阶段三联动】渐进训练期间, 层级上限受 coarse_index 限制 ==> Anchor级别LOD筛选
        #
        #   2. prefilter_voxel()  → 用gsplat投影, 做视锥剔除(radii>0),Anchor级别,视锥剔除
        #   3. generate_neural_gaussians() → MLP解码锚点特征→高斯属性
        #                           【阶段五】若 appearance_dim>0, 拼接外观码后输入Color MLP(筛选有效的GS)
        #                           【阶段二】若 dist2level=="progressive", 对临界层opacity做平滑插值
        #   4. gsplat.rasterization() → 高斯光栅化出图像

        # get_render_func() 根据 base_model 返回 "render"(3dgs/Scaffold-gs) 或 "render_2dgs"(2dgs)
        # viewpoint_cam: Camera对象(当前训练帧相机,包含相机世界坐标、W2C、FoVx、FoVy、H、W、resolution)
        # gaussians:高斯模型对象(pc),包含所有Anchor位置、特征、MLP网络权重等等
        render_pkg = getattr(modules, get_render_func(dataset.base_model))(viewpoint_cam, gaussians, pipe, scene.background, iteration, dataset.render_mode)
        # images是渲染出来的图像(后续与GT做L1_Loss),scaling是高斯缩放
        image, scaling = render_pkg["render"], render_pkg["scaling"]

        # ======================================================================
        # 损失函数计算
        # ======================================================================
        gt_image = viewpoint_cam.original_image.cuda()  # 真值图像
        
        # 主损失 = (1-λ)*L1 + λ*SSIM, 默认 lambda_dssim=0.2
        Ll1 = l1_loss(image, gt_image)                  # L1_Loss
        ssim_loss = (1.0 - ssim(image, gt_image))       # SSIM_Loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss    # L1_Loss与SSIM_Loss加权求和
       
        # 缩放正则化: 惩罚高斯体积过大, 默认 lambda_dreg=0.01
        # 作用: 防止高斯过度膨胀(GS体积过大), 保持模型紧凑(用更多小GS精细化描述表面)
        if opt.lambda_dreg > 0:
            if scaling.shape[0] > 0:
                scaling_reg = scaling.prod(dim=1).mean()
            else:
                scaling_reg = torch.tensor(0.0, device="cuda")
            loss += opt.lambda_dreg * scaling_reg

        # 法线一致性损失 (仅2DGS变体使用, lambda_normal默认0.0)
        if opt.lambda_normal > 0 and iteration > opt.normal_start_iter:
            normals = render_pkg["render_normals"].squeeze(0).permute((2, 0, 1))
            normals_from_depth = render_pkg["render_normals_from_depth"] * render_pkg["render_alphas"].squeeze(0).detach()
            if len(normals_from_depth.shape) == 4:
                normals_from_depth = normals_from_depth.squeeze(0)
            normals_from_depth = normals_from_depth.permute((2, 0, 1))
            normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
            loss += opt.lambda_normal * normal_error.mean()

        # 深度畸变损失 (仅2DGS变体使用, lambda_dist默认0.0)
        if opt.lambda_dist and iteration > opt.dist_start_iter:
            loss += opt.lambda_dist * render_pkg["render_distort"].mean()
    
        # 反向传播 — 梯度会流回到 viewspace_points(means2d) 上
        # 这些梯度随后被 training_statis() 收集, 用于致密化决策
        loss.backward()
        # loss反向传播优化的参数: Anchor级别: 位置固定(xyz为对应Voxel中心)、Offset(Anchor所管辖的GS相对偏移)、_anchor_feat(Anchor的特征)、scaling(Anchor的缩放参数,前3维控制GS活动范围,后3维送入MLP预测Covariance)
        #                       MLP相关权重(有MLP_Opacity、MLP_Color、MLP_Cov)

        iter_end.record()

        with torch.no_grad():
            # ---------- 进度条更新 ----------
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # ---------- 日志/保存/测试 ----------
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, getattr(modules, get_render_func(dataset.base_model)), (pipe, scene.background, iteration, dataset.render_mode), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # ==================================================================
            # 【Octree-GS 阶段四: 自适应控制 — 致密化(生长+剪枝)】
            # 在 start_stat < iteration < update_until 期间执行:
            # (默认: start_stat=500, update_from=1500,update_until=25000,update_interval=100)
            # start_stat:开始收集统计量的迭代步数,默认500
            # update_from:开始执行致密化+剪枝迭代步数,默认1500
            # update_until:停止执行致密化+剪枝的迭代步数,默认250000
            # update_interval:执行生长+剪枝的迭代间隔,每间隔100次迭代执行一次Anchor增长+剪枝
            #
            # 与渐进性增长相结合:1500-10K只允许同层Anchor点增长,10K后允许跨层增长
            # 渐进性增长会限制LOD最大层级,逐步解锁最大层级
            # 
            # 1500-10K 只允许同层Anchor点增长
            # 10K-25K 允许跨层增长+同级增长,但是收到渐进性训练最大LOD层数限制,期间逐步解锁最大LOD层级
            # 25K-40K 单纯地进行优化,不进行Anchor点的生长+剪枝,Anchor数量完全固定

            # ==================================================================
            # =========================(迭代次数满足进行致密化+剪枝)==================
            # 致密化与剪枝使用不同GS属性,在不同筛选粒度下收集统计量
            # 1、致密化:依据GS在2D屏幕空间的位置梯度,并且只有经过光栅化(真正参与着色的GS)才会累积2D屏幕的位置梯度
            #          因为梯度来自loss.backward() 通过光栅化计算图反传,一个GS如果没被光栅化(radii=0),它的means2d.grad就是无效值,不能用。
            #          计数器:该GS被光栅化着色的次数
            #
            # 2、剪枝:依据Anchor所管辖的有效GS的透明度(opacity)总和,只要MLP认为的Opacity>0的GS都会被累积透明度,
            #         将Anchor所管辖的M个有效GS(M<=K)进行不透明度累加,Opacity也需要逐帧累加,目的是统计这个 Anchor 在多个视角下的平均不透明度贡献
            #         计数器:该 Anchor 被观测到的帧数
            #         example:一个 Anchor 被观测了 80 帧，但 opacity 累积才 0.2 ==>需要剪枝
            if iteration < opt.update_until and iteration > opt.start_stat:
                
                # (1) 每步收集三类统计量 (basic_model.py: training_statis)
                #     a. 光栅化GS的2D屏幕位置梯度 → 在GS层面累积2D位置梯度 offset_gradient_accum [N*k] (生长依据)
                #     b. 每帧将Anchor管辖的有效GS(单纯MLP>0 不管是否被光栅化)的opacity求和后累加到 opacity_accum [N] (剪枝依据)
                #     c. Anchor被观测帧数 +1 → anchor_demon [N]               (归一化分母,若Anchor在该帧被观察到)
                gaussians.training_statis(render_pkg, image.shape[2], image.shape[1])
                
                # (2) 每 update_interval(默认100)步执行一次致密化+剪枝
                #     run_densify() 内部流程 (lod_model.py):
                #     
                #     a) anchor_growing() — 生长新锚点:
                #        先计算每个GS的平均梯度 avg_grad = offset_gradient_accum / offset_denom(只有在每帧被光栅化的Anchor才会累积2D位置梯度)
                #        遍历每一层 cur_level(不同LOD层级):
                #        - 计算动态阈值 τ_L = τ_g × (fork^β)^L                     (τ_g=0.0002, β=update_ratio=0.2)
                #        - 同层生长: τ_L ≤ avg_grad < τ_{L+1} → 在当前层添加锚点
                #        - 跨层生长: avg_grad ≥ τ_{L+1} → 在L+1层添加锚点       ⚠ 仅在渐进训练结束后(iteration > coarse_intervals[-1])才允许!
                #        - ΔL更新: anchor级平均梯度 > τ_L×extra_ratio(0.25) → _extra_level += extra_up(0.02) ⚠同样仅在渐进训练结束后才执行
                #        - 每次新增锚点都要经过 weed_out() 可见性检查,LOD层级能被大部分的相机看见,不一定要在相机的视锥体以内
                #     
                #     b) prune_anchor() — 剪枝低贡献锚点:
                #        - 条件: opacity_accum < min_opacity(0.005) × anchor_demon  (累积时需要Anchor在视锥体以内,并且通过LOD筛选,累积的GS只需满足Opacity>0,不需要光栅化)
                #          且 anchor_demon > update_interval × success_threshold(0.8)
                #        - 即: Anchor被足够多次观测但平均不透明度仍然很低的锚点会被剪枝删除
                if opt.densification and iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.run_densify(iteration, opt)
            
            # 致密化+剪枝阶段结束后, 清理临时统计量
            elif iteration == opt.update_until:
                gaussians.clean()
                    
            # ---------- 优化器步进(每次迭代都进行,与是否致密化+剪枝无关) ----------
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
            # ---------- 保存Checkpoint(每隔一定迭代步保存一次,默认1000) ----------
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    """初始化输出目录和TensorBoard写入器。
    如果未指定 model_path, 自动用UUID生成。
    """
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))

    # 保存配置参数:将所有的配置参数转换为字符串,保存到cfg_args文件中,方便查看
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 初始化TensorBoard写入器,训练过程通过tb_writer.add_scalar() 记录 loss、PSNR 等曲线
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    """训练过程中的日志记录 + 周期性测试集评估。
    在 testing_iterations 指定的迭代步进行测试集和训练集子集的评估。
    """
    # 1) 若使用TensorBoard,每步记录训练指标(每次调用都执行)
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)           # L1 Loss损失
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)       # Ltotal 损失
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)                               # 记录每步耗时

    # 同步到Wandb
    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # 2) 周期性测试集评估(在 testing_iterations 指定的迭代步执行) 例如[10k,20K,30K,40K]
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        # 构建两组评估数据: 测试集(全部) + 训练集子集(每隔5个取一个,5张)    
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        # ========== 外层 for: 依次评估 test 和 train 两组配置 ==========
        # 第1轮: config['name']='test',  cameras=全部测试集相机
        # 第2轮: config['name']='train', cameras=训练集中每隔5个取1个(共5张,用于快速监控训练集拟合情况)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0       # 累计L1损失(后续取平均)
                psnr_test = 0.0     # 累计PSNR(后续取平均)
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                # ========== 内层 for: 逐视角渲染 + 计算指标 ==========
                for idx, viewpoint in enumerate(config['cameras']):
                    
                    # 用当前模型渲染该视角,clamp到[0,1]防止越界,渲染该视角的Image,取该视角下的GT
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # 将前30个视角的渲染图和误差图写入TensorBoard(太多会占满磁盘)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                        
                        # GT图只在第一次评估时写入(GT不变,避免重复写入)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    # 累计每个视角的L1和PSNR指标
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                # ========== 内层 for 结束: 计算平均指标并输出 ==========
                psnr_test /= len(config['cameras'])     # 所有视角的平均PSNR
                l1_test /= len(config['cameras'])       # 所有视角的平均L1
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))   # 输出第几轮迭代,测试的性能PSNR SSIM的值为多少
                
                # 将平均指标写入TensorBoard/wandb
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        # ========== 外层 for 结束: 记录当前Anchor总数 + 清理显存 + 恢复训练模式 ==========
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', len(scene.gaussians.get_anchor), iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()  # 从eval模式切回train模式,继续训练

def render_set(base_model, model_path, name, iteration, views, gaussians, pipe, background, render_mode):
    """渲染一组视角并保存图像: 训练的产物是一个3D Gaussian场景模型(一堆Anchor+MLP权重)
    为进行3D Model重建质量的评估,给定一个相机位姿(位置+朝向+内参),3D Model会根据相机位姿渲染出一张图片,然后将渲染图保存到磁盘中。

    输出: 渲染图/误差图/GT图 + 每视角可见高斯计数。
    【注意】渲染时会调用与训练相同的LOD选择流程(set_anchor_mask等)。
    
    参数:
        base_model:   str, 渲染后端 ("scaffoldgs"/"3dgs"/"2dgs")
        model_path:   str, 输出根目录 (如 outputs/mipnerf360/Bicycle/2026-03-03_18-00-00)
        name:         str, "test" 或 "train", 决定输出子目录名
        iteration:    int, 当前迭代步 (用于输出子目录命名, 如 ours_40000)
        views:        list[Camera], 要渲染的相机列表
        gaussians:    GaussianLoDModel, 训练好的高斯模型
        pipe:         PipelineParams
        background:   [3] 背景颜色
        render_mode:  str, "RGB" 或 "RGB+ED"
    
    返回:
        t_list:             list[float], 每个视角的渲染耗时(秒), 用于计算FPS
        visible_count_list: list[int],   每个视角参与渲染的GS数量
    """
    # ---------- 创建输出目录 ----------
    # 目录结构: model_path/test/ours_40000/{renders, errors, gt}/
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []                 # 每个视角的渲染耗时
    visible_count_list = []     # 每个视角参与渲染的GS数量
    per_view_dict = {}          # {文件名: 可见GS数} 用于保存到JSON
    modules = __import__('gaussian_renderer')
    
    # ---------- 逐视角渲染循环(可能有多组相机位姿,逐个位姿遍历) ----------
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        # 精确计时: synchronize确保GPU操作完成后再取时间戳
        torch.cuda.synchronize();t_start = time.time()
        render_pkg = getattr(modules, get_render_func(base_model))(view, gaussians, pipe, background, iteration, render_mode)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)  # 记录该视角渲染耗时

        # 渲染结果: RGB渲染图像 + 可见GS数量
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)     # 光栅化输出的RGB图像 [3, H, W],每个Pixel都是由α-Blending公式计算得到
        visible_count = render_pkg["visibility_filter"].sum()       # 该视角参与渲染的GS数量(radii>0) 是经过LOD+视锥+MLP有效筛选+参与光栅化的GS个数
        visible_count_list.append(visible_count)

        # GT图像
        gt = view.original_image[0:3, :, :]     # [3, H, W]
        
        # 误差图: |渲染图 - GT| 的逐像素绝对差 ==> 渲染偏差图
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        # 渲染误差图,误差图亮度=渲染偏差大小,越暗,像素值越接近0,Render效果与GT几乎一致
        #           误差图越亮=渲染偏差越大,Render效果与GT差异大,该区域重建质量差
        errormap = (rendering - gt).abs()  

        # 保存三种图像到对应目录 (文件名格式: 00000.png, 00001.png, ...)
        # render渲染图、render误差图、GT图
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    # 保存每个视角的可见GS数到JSON文件 (用于后续evaluate()分析)
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset, opt, pipe, iteration, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    """渲染训练集/测试集并计算FPS。
    注意: 会重新加载模型并调用 set_coarse_interval() 以确保渲染时LOD选择正确。
    
    与 training_report() 的区别:
      - training_report(): 训练过程中周期性调用,用当前训练中的模型渲染,得到阶段性3D重建性能指标,只写TensorBoard
      - render_sets():     训练结束后调用,重新加载checkpoint,渲染结果保存为PNG到磁盘,用于后续evaluate()计算PSNR/SSIM/LPIPS
    
    参数:
        dataset:     ModelParams, 数据集配置
        opt:         OptimizationParams, 优化参数(需要coarse_iter等用于LOD设置)
        pipe:        PipelineParams
        iteration:   int, 要加载的checkpoint迭代步
        skip_train:  bool, 是否跳过训练集渲染
        skip_test:   bool, 是否跳过测试集渲染
    
    返回:
        visible_count: list[int], 最后一组渲染的每视角可见GS数量
    """
    with torch.no_grad():  # 推理阶段不需要梯度
        # ---------- 重新加载模型 ----------
        # 不复用训练中的模型,而是从checkpoint重新加载,确保状态干净
        modules = __import__('scene.gs_model_'+dataset.base_model, fromlist=[''])
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])  # 创建新的空模型
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)  # 加载checkpoint权重
        gaussians.eval()               # 切换到eval模式(MLP关闭dropout等)
        gaussians.set_coarse_interval(opt)  # 设置LOD渐进训练时间表(coarse_intervals会决定只渲染已解锁的层级，避免渲染尚未训练的层,一般训练结束后所有LOD层都解锁)
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        # ---------- 渲染训练集(render_sets()渲染+FPS计算) ----------
        if not skip_train:
            # 根据相机位姿(位置+朝向+内参) ==> 3DGS会渲染出图像,并计算该视角下参与渲染的GS数量以及渲染耗时以计算FPS(FPS = 1/平均渲染耗时)
            t_train_list, visible_count = render_set(dataset.base_model, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, dataset.render_mode)
            # 计算FPS: 跳过前5帧(GPU预热,首次渲染会触发CUDA kernel编译,耗时偏高)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')  # 紫色高亮输出
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/train_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        # ---------- 渲染测试集(与训练集渲染同理) ----------
        if not skip_test:
            t_test_list, visible_count = render_set(dataset.base_model, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, dataset.render_mode)
            # 同样跳过前5帧计算FPS
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    # 返回最后一组(test/train)的可见GS数量列表
    # visible_count是一个list,每个元素是单个视角下的可见GS数量,不是平均值
    # 比如我有N张图片,就有N个相机位姿,进行N个视角的渲染,得到N个可见GS数量的列表
    return visible_count  

#   把render_set()之前保存到磁盘的PNG图片重新读回GPU Tensor中,用于计算PSNR/SSIM/LPIPS等指标
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    # 遍历renders_dir目录下的所有PNG文件
    for fname in os.listdir(renders_dir):   
        render = Image.open(renders_dir / fname)    # 用PIL打开渲染图 (H,W,3)
        gt = Image.open(gt_dir / fname)             # 用PIL打开GT图 (H,W,3)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())  # 将渲染图转换为GPU Tensor (1,3,H,W)
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())          # 将GT图转换为GPU Tensor (1,3,H,W)
        image_names.append(fname)                   # 记录文件名
    return renders, gts, image_names


def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None, source_path=None, scene_name=None, base_layer=None):
    """对渲染结果计算质量指标: PSNR, SSIM, LPIPS, 以及平均可见高斯数。
    结果保存到 results.json 和 per_view.json。
    """

    full_dict = {}              # 存整个场景重建性能的平均指标,写入results.json
    per_view_dict = {}          # 存每个视角的重建性能指标,写入per_view.json
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name  # 定位渲染结果目录+读取图片(L614-626),如outputs/.../test/

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)  # 读取渲染图和GT图 ====> renders:N个[1,3,H,W], gts:N个[1,3,H,W]

        # 评估三维重建质量的性能指标:
        #   3D重建效果的好坏,关键看给定一个模型未见过的相机位姿,其渲染出来的图片与GT图是否接近
        #   Dataset通常是真实世界拍摄的一组图片,根据Colmap去估计相机位姿,重建点云,划分训练集和测试集
        #   用训练集的相机位姿进行场景重建,优化GS分布以及属性
        #   用测试集的相机位姿进行场景渲染,得到测试集的渲染图和GT图,然后计算PSNR,SSIM,LPIPS等指标
        ssims = []
        psnrs = []
        lpipss = []

        # 逐个视角计算三大指标:PSNR, SSIM, LPIPS
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))                 # SSIM: 结构相似性,值域[0,1],越大越好
            psnrs.append(psnr(renders[idx], gts[idx]))                 # PSNR: 峰值信噪比(dB),值域[0,+∞),越大越好
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())   # LPIPS: 感知距离,值域[0,+∞),越小越好
        
        # 输出模型路径,所有视角的平均PSNR SSIM LPIPS 以及平均每帧可见的GS数量 
        logger.info(f"scene_name:  \033[1;35m{scene_name}\033[0m")
        logger.info(f"source_path: \033[1;35m{source_path}\033[0m")
        logger.info(f"model_path:  \033[1;35m{model_paths}\033[0m")
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        logger.info("  GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(visible_count).float().mean(), ".5"))
        print("")

        # 将评估的结果写入Wandb绘图
        if wandb is not None:
            wandb.log({"test_PSNR":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_SSIM":torch.stack(ssims).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })
            wandb.log({"test_GS_NUMS":torch.stack(visible_count).float().mean().item(), })

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/GS_NUMS', torch.tensor(visible_count).float().mean().item(), 0)
        
        # 将测试集评估的结果保存到JSON文件中 (包含场景信息和路径)
        full_dict[scene_dir][method].update({
            "scene_name": scene_name if scene_name else "",
            "source_path": source_path if source_path else "",
            "model_path": str(model_paths),
            "base_layer": base_layer if base_layer is not None else "",
            "PSNR": torch.tensor(psnrs).mean().item(),
            "SSIM": torch.tensor(ssims).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
            "GS_NUMS": torch.tensor(visible_count).float().mean().item(),
            })

        # 将每个视角的评估结果保存到JSON文件中
        # eg:"PSNR":    {"00000.png": 28.1, "00001.png": 26.9, ...},  # 每张图各自的PSNR
        #    "SSIM":    {"00000.png": 0.9, "00001.png": 0.85, ...}, # 每张图各自的SSIM
        #    "LPIPS":   {"00000.png": 0.1, "00001.png": 0.15, ...}, # 每张图各自的LPIPS
        #    "GS_NUMS": {"00000.png": 100, "00001.png": 120, ...} # 每张图各自的GS数量
        per_view_dict[scene_dir][method].update({
            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            "GS_NUMS": {name:vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}
            })
    # 最后将测试集结果保存在outputs/.../results.json(平均指标)和per_view.json(每个视角的指标)中
    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging
    # 设置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

# ========================================================================
# 主入口Main函数
# ========================================================================
if __name__ == "__main__":
    # ---------- 命令行参数 ----------
    parser = ArgumentParser(description="Training script parameters")         # YAML配置文件路径:config/scaffoldgs/base_model.yaml 或者lod_model.yaml
    parser.add_argument('--config', type=str, help='train config file path')  # 核心: YAML配置文件路径(如base_model.yaml或lod_model.yaml),决定了使用哪个模型
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)       # Warmup模式: 训练两轮, 第二轮用第一轮输出的点云进行重新初始化
    parser.add_argument('--use_wandb', action='store_true', default=False)    # 是否使用wandb进行数据记录、绘制图表
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])  # 在哪些迭代步数进行测试集评估,比如[10K,20K,30K,40K]显示PSNR SSIM指标
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])  # 在哪些迭代步数保存.ply点云文件
    parser.add_argument("-m", "--model_path", type=str, default=None, help="自定义模型输出目录, 不指定则自动生成")
    parser.add_argument("-s", "--source_path", type=str, default=None, help="数据集路径, 不指定则使用config中的source_path")
    parser.add_argument("--quiet", action="store_true")  
    parser.add_argument("--extra_lod", type=int, default=0, help="自动计算LOD层数K后额外增加N层, 默认0")
    parser.add_argument("--base_layer", type=int, default=None, help="八叉树基础层号, 决定最粗体素大小, 不指定则使用config中的值")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 保存CheckPoint(完整模型状态的迭代步数)
    parser.add_argument("--start_checkpoint", type=str, default = None)              # 从哪个CheckPoint文件恢复训练
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    
    # ---------- 解析YAML配置文件(Octree-AnyGS的核心参数保存在YAML文件中) ----------
    # 配置文件包含三组参数(根据base_model.yaml或lod_model.yaml相应加载):
    #   lp (model_params):  模型类型, LOD参数(fork, levels, dist_ratio...), 场景路径
    #   op (optim_params):  迭代次数, 学习率, 致密化参数(τ_g, β, ω...)
    #   pp (pipeline_params): Pipeline设置
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        args.save_iterations.append(op.iterations)

    # 将命令行的 --extra_lod 注入到模型配置中
    if args.extra_lod != 0:
        lp.model_config['kwargs']['extra_lod'] = args.extra_lod

    # 命令行 --base_layer 覆盖 config 中的 base_layer
    if args.base_layer is not None:
        lp.model_config['kwargs']['base_layer'] = args.base_layer

    # 命令行 -s 覆盖 config 中的 source_path
    if args.source_path:
        lp.source_path = args.source_path

    # 构建Output文件路径并备份config.yaml
    if args.model_path:
        lp.model_path = args.model_path
    else:
        cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        lp.model_path = os.path.join("outputs", lp.dataset_name, lp.scene_name, cur_time)
    os.makedirs(lp.model_path, exist_ok=True)
    # 将实际使用的 source_path 和 model_path 写入保存的config.yaml
    cfg['model_params']['source_path'] = lp.source_path
    cfg['model_params']['model_path'] = lp.model_path
    with open(os.path.join(lp.model_path, "config.yaml"), 'w', encoding='utf-8') as f_cfg:
        yaml.dump(cfg, f_cfg, allow_unicode=True, default_flow_style=False)

    logger = get_logger(lp.model_path)

    # 设置默认的测试/保存时机(每10000步一次) test_iterations = [10000, 20000, 30000, 40000]
    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != op.iterations:
        args.test_iterations.append(op.iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != op.iterations:
        args.save_iterations.append(op.iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    # 备份实验的代码(项目文件.py .sh .yaml...)
    try:
        saveRuntimeCode(os.path.join(lp.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
    
    exp_name = lp.scene_name if lp.dataset_name=="" else lp.dataset_name+"_"+lp.scene_name
    
    # 若使用wandb,初始化项目,上传实验参数进行记录
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            project=f"Octree-GS",
            name=exp_name,
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + lp.model_path)

    # 初始化随机数种子(保证实验可复现)
    safe_state(args.quiet)

    # 启动GUI服务器
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # ======================================================================
    #                           执行训练(Octree-AnyGS的核心训练函数)
    # ======================================================================
    training(lp, op, pp, exp_name, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    
    # ---------- Warmup模式: 用第一轮输出的点云重新初始化八叉树再训练一轮 ----------
    # 这相当于用第一轮的结果作为更好的初始化, 可以提升最终质量
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(op.model_path, f'point_cloud/iteration_{op.iterations}', 'point_cloud.ply')
        training(lp, op, pp, exp_name, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger, new_ply_path)

    logger.info("\nTraining complete.")

    # ======================================================================
    # 渲染 + 评估 若eval=True => 只渲染+评估测试集; 
    #            若eval=False => 训练集渲染 + 评估(没有测试集)
    # ======================================================================
    logger.info(f'\nStarting Rendering~')
    if lp.eval:
        visible_count = render_sets(lp, op, pp, -1, skip_train=True, skip_test=False, wandb=wandb, logger=logger)
    else:
        visible_count = render_sets(lp, op, pp, -1, skip_train=False, skip_test=True, wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    logger.info("\n Starting evaluation...")
    eval_name = 'test' if lp.eval else 'train'
    # 读取Render的Image 以及 GT 计算PSNR/SSIM/LPIPS ==>将评估的结果保存到Result.json当中
    evaluate(lp.model_path, eval_name, visible_count=visible_count, wandb=wandb, logger=logger, source_path=lp.source_path, scene_name=exp_name, base_layer=lp.model_config['kwargs'].get('base_layer'))
    logger.info("\nEvaluating complete.")
