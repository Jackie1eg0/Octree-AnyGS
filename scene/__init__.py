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
scene/__init__.py — 场景管理器

Scene 类是整个 Octree-GS / Scaffold-GS 的数据管理中枢, 负责:
  1. 加载场景数据 (COLMAP / Blender / City 格式)
  2. 构建训练/测试相机列表
  3. 初始化高斯模型 (调用 create_from_pcd 或从 checkpoint 恢复)
  4. 提供模型保存/加载接口

┌──────────────────────────────────────────────────────────────────────┐
│                    Scene.__init__() 初始化流程                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 设置背景颜色 (黑/白/随机)                                        │
│                       ↓                                              │
│  2. 检查是否从checkpoint加载                                         │
│                       ↓                                              │
│  3. 根据数据格式调用对应的场景读取器                                  │
│     (Colmap / Blender / City)                                        │
│     → 获取 scene_info (点云 + 相机列表 + 场景范围)                   │
│                       ↓                                              │
│  4. 初始化外观嵌入 set_appearance()                                  │
│                       ↓                                              │
│  5. 降采样并保存初始点云 save_ply()                                  │
│     + 导出 cameras.json                                              │
│                       ↓                                              │
│  6. 构建训练/测试相机 Camera 对象列表                                │
│     (按 resolution_scale 分组)                                       │
│                       ↓                                              │
│  7a. (首次训练) create_from_pcd() → 八叉树初始化                     │
│  7b. (恢复训练) load_ply() + load_mlp_checkpoints()                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

在 train.py 中的典型调用:
    scene = Scene(dataset, gaussians, logger=logger)
    → 内部完成了从"原始数据"到"可训练高斯模型"的全部初始化
"""

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import BasicPointCloud
import numpy as np


class Scene:
    """场景管理器: 数据加载、相机管理、模型初始化/保存/恢复。
    
    核心属性:
        model_path:        str, 模型输出目录 (保存checkpoint/PLY/cameras.json)
        gaussians:         GaussianLoDModel 或 GaussianModel, 高斯模型对象
        train_cameras:     dict, {resolution_scale: [Camera]} 训练相机列表
        test_cameras:      dict, {resolution_scale: [Camera]} 测试相机列表
        cameras_extent:    float, 场景范围半径 (用于 spatial_lr_scale)
        background:        [3] GPU张量, 背景颜色
        resolution_scales: list, 分辨率缩放因子列表 (如 [1.0])
        loaded_iter:       int 或 None, 如果从checkpoint恢复则为对应迭代步
    """

    def __init__(self, args, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, logger=None):
        """Scene 初始化: 加载场景数据 + 构建相机 + 初始化/恢复高斯模型。
        
        参数:
            args:              命令行参数对象, 包含:
                - source_path:       str, 数据集根目录 (COLMAP/Blender场景路径)
                - model_path:        str, 模型输出目录
                - images:            str, 图像子目录名 (默认 "images")
                - eval:              bool, 是否划分训练/测试集
                - white_background:  bool, 是否使用白色背景
                - random_background: bool, 是否使用随机背景
                - ratio:             int, 点云降采样比例 (每ratio个点取1个,默认1全采样)
            gaussians:         GaussianLoDModel (Octree-GS) 或 GaussianModel (Scaffold-GS)
                               高斯模型对象, 此时尚未初始化参数
            load_iteration:    int 或 None, 从哪个checkpoint恢复:
                - None:  首次训练, 从SfM点云初始化
                - -1:    自动查找最新的checkpoint
                - >0:    指定具体迭代步
            shuffle:           bool, 是否随机打乱相机顺序 (默认True)
            resolution_scales: list, 分辨率缩放因子列表 (默认[1.0])
                               如 [1.0, 0.5] 表示同时使用原始分辨率和半分辨率
            ply_path:          str 或 None, 外部提供的PLY点云路径 (覆盖数据集自带的点云)
            logger:            日志记录器
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scales = resolution_scales
    
        # =====================================================================
        # Step 1: 设置背景颜色
        # =====================================================================
        # 背景颜色在光栅化时用于填充没有Gaussian覆盖的像素
        # Blender合成数据集通常使用白色背景, 真实场景使用黑色
        if args.random_background:
            self.background = torch.rand(3, dtype=torch.float32, device="cuda")     # 随机颜色 (数据增强)
        elif args.white_background:
            self.background = torch.ones(3, dtype=torch.float32, device="cuda")     # 白色 [1,1,1]
        else:
            self.background = torch.zeros(3, dtype=torch.float32, device="cuda")    # 黑色 [0,0,0]

        # =====================================================================
        # Step 2: 检查是否从 checkpoint 恢复
        # =====================================================================
        if load_iteration:
            if load_iteration == -1:
                # 自动查找 model_path/point_cloud/ 下最大的 iteration_XXXX 目录
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # train_cameras / test_cameras 存储格式:
        #   dict: {resolution_scale: [Camera, Camera, ...]}
        #   key   = float, 分辨率缩放因子 (如 1.0)
        #   value = list of Camera (nn.Module), 每个 Camera 对象包含:
        #     ─── 基本信息 ───
        #       uid:                  int,              相机顺序编号
        #       colmap_id:            int,              原始 CameraInfo 的 uid
        #       image_name:           str,              图像名称 (不含扩展名)
        #       resolution_scale:     float,            分辨率缩放因子
        #     ─── 内外参 ───
        #       R:                    [3,3] ndarray,    旋转矩阵 (W2C旋转的转置, CUDA glm约定)
        #       T:                    [3] ndarray,      平移向量 (W2C的平移部分)
        #       FoVx:                 float,            水平视场角 (弧度)
        #       FoVy:                 float,            垂直视场角 (弧度)
        #     ─── 图像数据 ───
        #       original_image:       [3, H, W] CUDA Tensor, GT图像 (已缩放, clamp到[0,1])
        #       image_width:          int,              缩放后的图像宽度
        #       image_height:         int,              缩放后的图像高度
        #     ─── 裁剪面 ───
        #       znear:                float,            近裁剪面 = 0.01
        #       zfar:                 float,            远裁剪面 = 100.0
        #     ─── 预计算的变换矩阵 (CUDA Tensor) ───
        #       world_view_transform: [4,4],            W2C矩阵 (列主序转置存储)
        #       projection_matrix:    [4,4],            透视投影矩阵
        #       full_proj_transform:  [4,4],            完整MVP矩阵 = W2C × Projection
        #       camera_center:        [3],              相机在世界坐标系中的位置
        #
        # train_cameras 与 test_cameras 结构完全相同, 区别仅在数据来源:
        #   COLMAP:  LLFF协议, 每llffhold(=8)张取1张做测试, 其余训练
        #   Blender: transforms_train.json → 训练, transforms_test.json → 测试
        #   City:    同COLMAP的LLFF协议划分
        #   eval=False 时, 所有相机归入训练集, 测试集为空列表
        self.train_cameras = {}     # {scale: [Camera]}
        self.test_cameras = {}      # {scale: [Camera]}
        
        # =====================================================================
        # Step 3: 根据数据格式调用对应的场景读取器
        # =====================================================================
        # 通过检查 source_path 下的特征性文件/目录来判断数据格式:
        #   - sparse/       → COLMAP 格式 (真实场景, 如 Mip-NeRF360)
        #   - transforms_train.json → Blender 格式 (合成场景, 如 NeRF Synthetic)
        #   - transforms.json       → City 格式
        #
        # 返回的 scene_info (SceneInfo 命名元组) 包含:
        #   - point_cloud:       BasicPointCloud (SfM点云: points, colors, normals)
        #   - train_cameras:     [CameraInfo] 训练相机信息列表,所含的信息格式如上注释
        #   - test_cameras:      [CameraInfo] 测试相机信息列表
        #   - nerf_normalization: dict, {"translate": [3], "radius": float} 场景归一化参数,自适应不同尺度的场景
        #   - ply_path:          str, 原始PLY点云路径
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, warmup_ply_path=ply_path)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.eval, warmup_ply_path=ply_path)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            scene_info = sceneLoadTypeCallbacks["City"](args.source_path, args.eval, warmup_ply_path=ply_path)
        else:
            assert False, "Could not recognize scene type!"

        # =====================================================================
        # Step 4: 初始化外观嵌入
        # =====================================================================
        # 为每个训练相机分配一个可学习的外观向量 (处理光照/白平衡差异)
        # 在 lod_model.py 中创建 Embedding(num_cameras, appearance_dim)
        self.gaussians.set_appearance(len(scene_info.train_cameras))
        
        # =====================================================================
        # Step 5: (首次训练时) 降采样点云 + 导出相机JSON
        # =====================================================================
        if not self.loaded_iter:
            # 5a. 降采样SfM点云并保存为 input.ply
            # args.ratio 控制降采样率: 每ratio个点取1个 (减少初始化时的计算量)
            pcd = self.save_ply(scene_info.point_cloud, args.ratio, os.path.join(self.model_path, "input.ply"))
            
            # 5b. 导出所有相机参数为 cameras.json (用于可视化和调试)
            # 导出为JSON格式,包含camera_id img_name width height(图像的W H) position(相机在World坐标系下的位置) rotation(相机的旋转矩阵) fx fy(焦距)
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # =====================================================================
        # Step 6: 构建 Camera 对象列表
        # =====================================================================
        if shuffle:
            # 打乱相机顺序, 确保训练时不按固定顺序遍历
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # cameras_extent: 所有相机中心的最大离散距离 (场景"半径")
        # 用于计算 spatial_lr_scale —— 控制位置学习率的大小,
        # 使得在大场景中位置学习率更大, 小场景中更小
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 按 resolution_scale 分组创建 Camera 对象
        # 每个 Camera 对象包含: 图像张量(缩放到目标分辨率), 内外参, 相机中心等
        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, self.background)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, self.background)

        # =====================================================================
        # Step 7: 初始化或恢复高斯模型
        # =====================================================================
        if self.loaded_iter:
            # 7b. 从 checkpoint 恢复: 加载PLY锚点数据 + MLP权重
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
        else:
            # 7a. 首次训练: 从降采样后的SfM点云创建八叉树高斯模型
            # create_from_pcd() 内部流程:
            #   1) set_level()     → 根据SFM点云和相机位置计算 d_max, 确定Octree的最大LOD层数K上限
            #   2) octree_sample() → 多层体素化, 构建八叉树,初始化时用同一批的SFM点云在不同层级的Voxel中心构建
            #   3) weed_out()      → 可见性裁剪,初始化生成的不同层级的Anchor点可能冗余,确保大概90%的相机能见到这个Anchor点才保留,剔除的主要是高层级+稀疏相机覆盖区域Anchor点
            #   4) 初始化所有可学习参数 (anchor/offset/feat/scaling/rotation)
            #
            # 这里传入 self.train_cameras 和 self.resolution_scales,
            # 用于 set_level() 计算相机-点云距离来确定LOD层数,构建Octree
            self.gaussians.create_from_pcd(pcd, self.cameras_extent, logger, self.train_cameras, self.resolution_scales)

    def save_ply(self, pcd, ratio, path):
        """降采样SfM点云并保存为PLY文件。
        
        SfM重建的点云通常非常密集(数十万~百万点), 直接用于初始化
        会导致体素化时间过长。通过降采样 (每ratio个点取1个) 减少计算量,
        同时保留足够的空间覆盖度用于八叉树初始化。
        
        参数:
            pcd:   BasicPointCloud, 原始SfM点云 (points, colors, normals)
            ratio: int, 降采样比例 (每ratio个点取1个, 如 ratio=1表示不降采样)
            path:  str, PLY文件保存路径 (通常为 model_path/input.ply)
        
        返回:
            BasicPointCloud, 降采样后的点云对象 (传递给 create_from_pcd)
        """
        new_points = pcd.points[::ratio]      # 每ratio个点取1个
        new_colors = pcd.colors[::ratio]
        new_normals = pcd.normals[::ratio]
        new_pcd = BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals)
        storePly(path, new_points, new_colors)  # 持久化到磁盘
        return new_pcd

    def save(self, iteration):
        """保存模型 checkpoint (PLY锚点数据 + MLP权重)。
        
        在 train.py 中按指定间隔调用 (如每7000步/30000步保存一次)。
        保存目录结构:
            model_path/point_cloud/iteration_XXXXX/
                ├── point_cloud.ply      — 锚点坐标/层级/特征/缩放等
                ├── opacity_mlp.pt       — 不透明度MLP权重
                ├── cov_mlp.pt           — 协方差MLP权重
                └── color_mlp.pt         — 颜色MLP权重
        
        参数:
            iteration: int, 当前训练迭代步 (用于构建目录名)
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), iteration)
        self.gaussians.save_mlp_checkpoints(point_cloud_path)
        
    def getTrainCameras(self):
        """获取所有分辨率下的训练相机列表。
        
        将 self.train_cameras 中不同 resolution_scale 的相机合并为一个扁平列表。
        在 train.py 的主循环中用于随机采样训练视角。
        
        返回:
            [Camera] 所有训练相机的列表每个Camera对象包含的信息包括 1)基本信息 2)内外参 3)图像数据 4)裁剪平面
        """
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.train_cameras[scale])
        return all_cams

    def getTestCameras(self):
        """获取所有分辨率下的测试相机列表。
        
        与 getTrainCameras() 对称, 在评估时使用。
        
        返回:
            [Camera] 所有测试相机的列表
        """
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.test_cameras[scale])
        return all_cams