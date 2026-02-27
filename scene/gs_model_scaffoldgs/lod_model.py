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
# GaussianLoDModel — Octree-GS 带LOD(Level-of-Detail)的高斯模型 (Scaffold-GS变体)
# ============================================================================
# 本文件实现了 Octree-GS 论文的核心模型类 GaussianLoDModel。
# 它继承自 BasicModel(basic_model.py), 在 Scaffold-GS 的基础上增加了:
#   - 八叉树(Octree)多层级锚点组织
#   - LOD(Level-of-Detail) 自适应渲染
#   - 渐进式训练(Coarse-to-Fine Progressive Training)
#   - 多层级自适应锚点生长与剪枝
#
# 【Octree-GS 五阶段在本文件中的对应】:
#   阶段一(八叉树构建与Anchor初始化):
#       set_level()       → 计算 d_max, d_min, 自动确定LOD层数K
#       octree_sample()   → 对点云进行多层体素化, 构建八叉树结构
#       create_from_pcd() → 完整的初始化流程(调用上述函数 + weed_out可见性裁剪)
#
#   阶段二(前向渲染与自适应LOD选择):
#       set_anchor_mask()            → 根据相机距离计算LOD等级, 筛选可见锚点
#       generate_neural_gaussians()  → MLP解码锚点特征 → 神经高斯属性(位置/颜色/不透明度/协方差)
#       map_to_int_level()           → 连续LOD等级 → 离散层级映射 (继承自BasicModel)
#
#   阶段三(渐进式训练: 从粗到精):
#       set_coarse_interval()  → 计算每层的解锁迭代步数(等比递减时间表)
#       set_anchor_mask()中的  → coarse_index 限制当前可用的最高层级
#
#   阶段四(自适应控制: 锚点生长与剪枝):
#       anchor_growing()  → 基于梯度的同层生长 + 跨层生长 + ΔL更新
#       run_densify()     → 调用 anchor_growing() + prune_anchor() 的完整致密化流程
#       prune_anchor()    → 基于不透明度的低贡献锚点剪枝
#       training_statis() → 累积梯度/不透明度统计量 (继承自BasicModel)
#
#   阶段五(外观特征嵌入):
#       embedding_appearance + mlp_color → 每相机外观编码, 处理光照/曝光差异
# ============================================================================

import os
import time
import torch
import math
import numpy as np
from torch import nn
from einops import repeat                                   # 张量重复操作(用于锚点→offset展开)
from functools import reduce
from torch_scatter import scatter_max                       # 分组聚合(生长时对同一体素的特征取max)
from utils.general_utils import get_expon_lr_func, knn      # 指数学习率衰减 + KNN邻居搜索
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement                     # PLY点云文件读写
from utils.graphics_utils import BasicPointCloud
from scene.embedding import Embedding                       # 外观嵌入码(per-camera appearance embedding)
from scene.basic_model import BasicModel                    # 基类: 提供优化器管理/梯度统计/去重/LOD映射等通用功能
    
class GaussianLoDModel(BasicModel):
    """
    Octree-GS 的核心模型类 (Scaffold-GS + LOD变体)。
    
    继承自 BasicModel, 新增:
    - 八叉树多层级锚点管理 (_level, _extra_level)
    - 渐进式训练调度 (coarse_intervals)
    - LOD感知的锚点掩码 (set_anchor_mask)
    - 多层级自适应生长/剪枝 (anchor_growing, run_densify)
    
    核心数据结构:
    - _anchor:       [N, 3]    锚点3D坐标 (八叉树体素中心)
    - _level:        [N, 1]    锚点所属的八叉树层级 (0=最粗, levels-1=最细)
    - _extra_level:  [N]       锚点的连续LOD偏移量 ΔL (用于细粒度LOD调整)
    - _offset:       [N, k, 3] 每个锚点的k个神经高斯偏移量 (k=n_offsets)
    - _anchor_feat:  [N, c]    锚点特征向量 (c=feat_dim, 输入给MLP)
    - _scaling:      [N, 6]    锚点缩放参数 (前3维=offset缩放, 后3维=高斯缩放基数)
    - _rotation:     [N, 4]    锚点旋转四元数 (当前未使用, 固定为单位四元数)
    """

    def __init__(self, **model_kwargs):
        """
        使用YAML配置文件中的参数初始化模型。
        
        参数 (通过 **model_kwargs 传入, 来自 YAML 的 model_config.kwargs):
            feat_dim (int):         锚点特征维度, 默认32
            n_offsets (int):        每个锚点生成的神经高斯数量, 默认10
            fork (int):             八叉树分叉因子, 默认2 (每层体素边长缩小为上一层的 1/fork),2*2*2=8
            levels (int):           LOD总层数K, -1表示自动计算
            init_level (int):       渐进训练的初始层级, -1表示自动设为 K/2
            dist_ratio (float):     距离分位数比例, 默认0.999 (用于计算 d_max, d_min)
            base_layer (int):       八叉树基础层(决定最粗体素大小), -1表示自动计算
            visible_threshold (float): 可见频率阈值, 低于此值的锚点被裁剪
            progressive (bool):     是否启用渐进式训练 (Coarse-to-Fine)
            dist2level (str):       连续距离→离散层级的映射方式 ('floor'/'round'/'ceil'/'progressive')
            use_feat_bank (bool):   是否使用Feature Bank MLP (多分辨率特征聚合)
            appearance_dim (int):   外观嵌入维度, 0表示不使用外观嵌入
            view_dim (int):         视角方向维度, 默认3 (观察方向xyz)
            extend (float):         场景包围盒扩展因子
            padding (float):        体素坐标偏移量(通常为0或0.5×voxel_size)
        """
        # 将YAML配置中的所有参数设为实例属性
        # 例如: self.feat_dim = 32, self.n_offsets = 10, self.fork = 2, ...
        for key, value in model_kwargs.items():
            setattr(self, key, value)
        
        # =====================================================================
        # 【可学习参数】— 核心几何数据结构 (阶段一初始化, 阶段四动态增减)
        # =====================================================================
        self._anchor = torch.empty(0)           # [N, 3]    锚点位置 (八叉树体素中心坐标)
        self._level = torch.empty(0)            # [N, 1]    锚点离散层级 L ∈ {0, 1, ..., K-1}
        self._extra_level = torch.empty(0)      # [N]       锚点连续LOD偏移 ΔL (训练中通过梯度更新)
        self._offset = torch.empty(0)           # [N, k, 3] 每个锚点的k个神经高斯位置偏移
        self._anchor_feat = torch.empty(0)      # [N, c]    锚点特征向量 (输入给MLP用于解码高斯属性)
        self._scaling = torch.empty(0)          # [N, 6]    缩放参数 (前3维: offset的缩放, 后3维: 高斯的缩放基数)
        self._rotation = torch.empty(0)         # [N, 4]    旋转四元数 (默认固定为[1,0,0,0], 不参与梯度更新)
        
        # =====================================================================
        # 【统计量】— 用于阶段四致密化决策 (在 training_statis 中累积)
        # =====================================================================
        self.opacity_accum = torch.empty(0)         # [N, 1]   各锚点的累积不透明度 (用于剪枝判断)
        self.anchor_demon = torch.empty(0)          # [N, 1]   各锚点被渲染的次数 (denominator, 用于归一化opacity)
        self.offset_gradient_accum = torch.empty(0) # [N*k, 1] 各神经高斯的累积viewspace梯度 (用于生长判断)
        self.offset_denom = torch.empty(0)          # [N*k, 1] 各神经高斯的梯度累积次数 (用于归一化梯度)
                
        self.optimizer = None
        self.spatial_lr_scale = 0       # 空间学习率缩放因子 (基于场景尺度调整位置/偏移的学习率)
        self.setup_functions()          # 初始化激活函数 (继承自BasicModel): exp, sigmoid, normalize
    
        # =====================================================================
        # 【MLP网络】— 阶段二/五: 将锚点特征解码为神经高斯属性
        # 所有MLP的输入 = [锚点特征(feat_dim) + 观察方向(view_dim)] (+ 外观码)
        # =====================================================================
        
        # Feature Bank MLP (可选): 实现多分辨率特征融合
        # 输入: 观察方向 [view_dim=3] → 输出: 3个权重 [3] (用softmax归一化)
        # 用于对不同频率的特征做自适应加权融合
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(self.view_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()
            
        # 不透明度MLP: 输入 [feat_dim + view_dim] → 输出 [n_offsets]
        # 每个锚点生成 k=n_offsets 个高斯的不透明度
        # 注意用Tanh而非Sigmoid: 输出范围[-1,1], 允许负值(ReLU阈值在后续处理)
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()
        
        # 协方差MLP: 输入 [feat_dim + view_dim] → 输出 [7 * n_offsets]
        # 每个高斯输出7维: 3维缩放(sigmoid激活) + 4维旋转四元数(normalize激活)
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).cuda()
    
        # 颜色MLP: 输入 [feat_dim + view_dim + appearance_dim] → 输出 [3 * n_offsets]
        # 【阶段五】如果 appearance_dim > 0, 额外拼接每相机的外观嵌入码
        # 用Sigmoid激活确保RGB值在[0,1]范围内
        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim+self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

    # ==========================================================================
    # 模型状态切换 (eval/train)
    # 渲染/评估时切换为eval模式 (关闭Dropout/BatchNorm等)
    # 训练时切换为train模式
    # ==========================================================================
    def eval(self):
        """将所有MLP网络切换到评估模式。在渲染/测试时调用。"""
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        """将所有MLP网络切换到训练模式。在训练循环中调用。"""
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()

    # ==========================================================================
    # 模型序列化 (capture/restore) — 用于Checkpoint保存和恢复训练
    # ==========================================================================
    def capture(self):
        """捕获当前模型完整状态, 用于保存Checkpoint。
        返回一个元组, 包含所有可学习参数、统计量和优化器状态。
        在 train.py 中通过 torch.save((gaussians.capture(), iteration), path) 保存。
        """
        param_dict = {}
        param_dict['optimizer'] = self.optimizer.state_dict()    # Adam优化器状态 (动量、二阶矩)
        param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
        param_dict['cov_mlp'] = self.mlp_cov.state_dict()
        param_dict['color_mlp'] = self.mlp_color.state_dict()
        if self.use_feat_bank:
            param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
        if self.appearance_dim > 0:
            param_dict['appearance'] = self.embedding_appearance.state_dict()
        return (
            self.voxel_size,            # 最粗层的体素大小
            self.standard_dist,         # 标准距离 d_max (LOD计算的归一化基准)
            self._anchor,               # 锚点坐标
            self._level,                # 锚点层级
            self._extra_level,          # 锚点LOD偏移 ΔL
            self._offset,               # 神经高斯偏移量
            self._scaling,              # 缩放参数
            self._rotation,             # 旋转参数
            self.opacity_accum,         # 不透明度累积量 (致密化统计)
            self.anchor_demon,          # 锚点渲染计数 (致密化统计)
            self.offset_gradient_accum, # 梯度累积量 (致密化统计)
            self.offset_denom,          # 梯度累积计数 (致密化统计)
            param_dict,                 # MLP权重 + 优化器状态
            self.spatial_lr_scale,      # 空间学习率缩放因子
        )
    
    def restore(self, model_args, training_args):
        """从Checkpoint恢复模型状态, 与 capture() 配对使用。
        先解包所有参数, 重新初始化优化器, 再加载优化器和MLP的权重。
        """
        (self.voxel_size,
        self.standard_dist,
        self._anchor,
        self._level,
        self._extra_level,
        self._offset,
        self._scaling,
        self._rotation,
        self.opacity_accum, 
        self.anchor_demon,
        self.offset_gradient_accum,
        self.offset_denom,
        param_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)       # 重建优化器结构
        self.optimizer.load_state_dict(param_dict['optimizer'])  # 恢复Adam动量/二阶矩
        self.mlp_opacity.load_state_dict(param_dict['opacity_mlp'])
        self.mlp_cov.load_state_dict(param_dict['cov_mlp'])
        self.mlp_color.load_state_dict(param_dict['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(param_dict['feature_bank_mlp'])
        if self.appearance_dim > 0:
            self.embedding_appearance.load_state_dict(param_dict['appearance'])

    # ==========================================================================
    # 属性访问器 (Properties)
    # 提供对内部参数的安全访问, 部分属性会经过激活函数处理
    # ==========================================================================
    @property
    def get_anchor(self):
        """返回锚点坐标 [N, 3], 直接返回原始参数。"""
        return self._anchor
    
    @property
    def get_level(self):
        """返回锚点层级 [N, 1], 整数值 0~K-1。"""
        return self._level
    
    @property
    def get_extra_level(self):
        """返回锚点LOD偏移量 ΔL [N], 浮点数, 训练中通过梯度微调。"""
        return self._extra_level
        
    @property
    def get_anchor_feat(self):
        """返回锚点特征 [N, feat_dim], 输入给MLP解码高斯属性。"""
        return self._anchor_feat

    @property
    def get_offset(self):
        """返回神经高斯偏移量 [N, n_offsets, 3]。"""
        return self._offset
    
    @property
    def get_scaling(self):
        """返回经过 exp() 激活的缩放参数 [N, 6] (确保正数)。"""
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        """返回经过 normalize 激活的旋转四元数 [N, 4] (确保单位长度)。"""
        return self.rotation_activation(self._rotation)

    def set_appearance(self, num_cameras):
        """【阶段五】初始化外观嵌入表。
        为每个训练相机分配一个可学习的外观向量, 用于处理光照/白平衡差异。
        在 Scene.__init__() 中调用。
        
        参数:
            num_cameras: 训练集相机数量
        """
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
        else:
            self.embedding_appearance = None

    @property
    def get_appearance(self):
        """返回外观嵌入表 (Embedding layer), 通过camera_id索引。"""
        return self.embedding_appearance
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    # ==========================================================================
    # 【阶段三】渐进式训练调度 — 计算每层的解锁时间表
    # ==========================================================================
    def set_coarse_interval(self, opt):
        """计算渐进训练(Coarse-to-Fine)的层级解锁时间表。
        Octree-GS 有 K 个层级(level 0 ~ level K-1),从最粗到最细。如果一开始就让所有层级都参与训练,粗层的全局结构还没收敛，细层锚点就开始贡献梯度，会导致训练不稳定。
        Coarse-to-Fine 渐进训练策略,逐步解锁更精细的层,为Coarse层分配更多的训练步数
        L0:分配 a1步数, L1:分配 a1*q步数, L2:分配 a1*q^2步数, ... q为1/1.5=0.667

        参数:
            opt: 优化参数, 包含:
                coarse_iter (int):   渐进训练总步数, 默认10000
                coarse_factor (float): 等比衰减因子 ω, 默认1.5
        
        示例 (K=6, init_level=3, coarse_iter=10000, ω=1.5):
          需要解锁的层数 = 6-1-3 = 2层 (第4层和第5层)
          q = 1/1.5 = 0.667, 等比数列之和 = 总步数Coarse_iter(10000) S = a₁ * (1 - qⁿ) / (1 - q)  n=2(所需解锁的层数)
          a1 = 10000*(1-0.667)/(1-0.667^2) = 6000
          a2 = a1*q = 4000
          0-6000步: 只训练L0-L3
          6000-10000步: 训练L0-L4(依此类推)
          
        """
        self.coarse_intervals = []                      # 解锁对应层次的迭代步数列表eg:[6000,10000]
        num_level = self.levels - 1 - self.init_level   # 需要逐步解锁的层数num_level=2
        if num_level > 0:
            q = 1/opt.coarse_factor                     # 等比公比 (默认 1/1.5 ≈ 0.667)
            a1 = opt.coarse_iter*(1-q)/(1-q**num_level) # 等比数列首项 (最粗层分配的步数)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval   # 累积迭代步数 (解锁阈值)
                temp_interval = interval
                self.coarse_intervals.append(interval)

    # ==========================================================================
    # 【阶段一-Step1】根据场景中相机-点云距离,确定Octree-GS需要多少层的LOD
    # ==========================================================================
    def set_level(self, points, cameras, scales):
        """根据相机-点云距离分布, 自动确定 d_max, d_min 和 LOD层数K。
        
        核心思想: LOD层数应该覆盖场景中"最远"到"最近"的观察距离范围。
        
        计算流程:
          1. 遍历所有相机, 计算每个相机位置到所有SFM点云(Colmap输出)的距离
          2. 取 dist_ratio 分位数(默认0.999)得到上述距离的d_max和d_min
          3. 对所有相机的d_max/d_min再取分位数, 得到全局的 standard_dist(d_max)
          4. LOD层数 K = round(log2(d_max/d_min) / log2(fork)) + 1
        
        参数:
            points:  [M, 3] 初始SfM点云坐标
            cameras: dict, {scale: [Camera]} 不同分辨率下的训练相机列表
            scales:  list, 分辨率缩放因子列表 (如 [1.0])
        
        设置的属性:
            self.standard_dist: 标准距离 d_max, LOD公式 L* = log2(d_max/d) 的常数
            self.levels:        LOD总层数K (如果YAML中设为-1则自动计算)
            self.init_level:    渐进训练的起始层 (如果YAML中设为-1则设为 K/2)
            self.cam_infos:     [num_cams, 4] 所有相机的 [cx, cy, cz, scale]
        """
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()       # 保存所有相机信息, 后续weed_out()使用
        for scale in scales:
            for cam in cameras[scale]:                          # 遍历该分辨率下的所有相机,for循环内为单个相机操作
                cam_center = cam.camera_center                  # 单个相机世界中心坐标(x,y,z)
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                # 计算某单个相机到所有SFM点云的距离,Points:[M,3]为SFM点云坐标,cam_center:[3]->[M, 3]通过广播
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                # 取分位数去除异常值 (dist_ratio=0.999 → 去掉最远0.1%和最近0.1%的点)
                dist_max = torch.quantile(dist, self.dist_ratio)
                dist_min = torch.quantile(dist, 1 - self.dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale                         # 乘以分辨率缩放因子,在不同分辨率下d_max d_min有差异
                all_dist = torch.cat((all_dist, new_dist), dim=0)   # 每个相机的d_max d_min拼接到一起
        
        # 对全部相机的距离统计量再取一次分位数, 得到全局 d_max, d_min
        dist_max = torch.quantile(all_dist, self.dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - self.dist_ratio)
        self.standard_dist = dist_max               # d_max: LOD公式的归一化常数
        if self.levels == -1:
            # 自动计算LOD层数: K = round(log_fork(d_max/d_min)) + 1  与论文公式5)一致
            # 设置中self.fork=2 即标准八叉树(2³=8子节点),(每层体素边长 = 上层/fork)
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        if self.init_level == -1:
            self.init_level = int(self.levels/2)     # 渐进训练初始层 = K/2 (最开始解锁L0-L(K/2-1)层的训练)
            
    # ==========================================================================
    # 【阶段一-Step2】八叉树多层体素化 — 构建多层级锚点结构
    # ==========================================================================
    def octree_sample(self, data):
        """对点云进行多层体素化, 构建八叉树结构。
        
        核心思想: 对同一组点云, 用不同粒度的体素网格进行量化:
          - Level 0: 体素大小 = voxel_size (最粗, 覆盖大区域)
          - Level 1: 体素大小 = voxel_size / fork (中等)
          - Level K-1: 体素大小 = voxel_size / fork^(K-1) (最细, 捕捉细节)
        
        每个层级独立体素化, 然后全部拼接在一起。
        这不是真正的树形结构, 而是"扁平化的八叉树": 所有层级的体素中心
        一起存储, 通过 _level 标签区分层级。
        
        参数:
            data: [M, 3] 输入SFM点云坐标
        
        设置的属性:
            self.positions: [N_total, 3] 所有层级的体素中心坐标
            self._level:    [N_total]    每个位置的层级标签
        """
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()   # 所有层级的anchor点坐标
        self._level = torch.empty(0).int().cuda()           # 对应层级标签
        for cur_level in range(self.levels):
            # 遍历Octree的每一层级0、1、2...K-1
            # 当前层的体素大小: voxel_size / fork^cur_level,层级越高,Voxel_Size越小。
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            # 体素化: 坐标量化到网格 → 去重(唯一体素) → 还原为世界坐标
            # round((SFM点坐标-init_pos)/当前VoxelSize) 得到整数网格坐标, unique去重, 再乘回当前VoxelSize
            # self.init_pos为场景包围盒中的左下后方顶点,定义一个从 init_pos开始的、均匀划分的体素网格坐标系
            new_positions = torch.unique(torch.round((data - self.init_pos) / cur_size), dim=0) * cur_size + self.init_pos  # 还原回世界坐标系+init_pos
            new_positions += self.padding * cur_size   # 加上padding偏移(默认为0,不偏移,anchor点在Voxel中心)
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level  # 给anchor点打上层级标签(Voxel经过去重后)
            
            # 把当前层新生成的Anchor点坐标[N_Cur,3]和层级标签[N_Cur,]拼接到self.positions和self._level中
            # 所有层级用的都是同一批 SfM 点云 data, 只是用不同粒度的Voxel去量化(同一批数据送入不同粒度网格进行初始化)
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    # ==========================================================================
    # 【阶段一-完整流程】从SfM点云创建八叉树高斯模型(剔除冗余Anchor点)+初始化所有可学习参数
    # ==========================================================================
    def create_from_pcd(self, pcd, spatial_lr_scale, logger, *args):
        """从SfM点云初始化完整的八叉树高斯模型。在 Scene.__init__() 中调用。
        
        完整流程:
          1. set_level()     → 计算 d_max, LOD层数K: 根据场景中相机-点云距离,确定Octree-GS层级深度
          2. 计算包围盒和体素大小
          3. octree_sample() → 多层体素化, 构建八叉树
          4. weed_out()      → 可见性裁剪, 去掉无法被任何相机看到的锚点
          5. 初始化所有可学习参数 (anchor/offset/feat/scaling/rotation)
        
        参数:
            pcd:              BasicPointCloud, SfM重建得到的初始点云
            spatial_lr_scale: 空间学习率缩放因子 (基于场景对角线长度)
            logger:           日志记录器
            *args:            传递给 set_level() 的 cameras, scales
        """
        points = torch.tensor(pcd.points).float().cuda()    # SFM点云坐标
        self.set_level(points, *args)                       # Step1: 确定 d_max, K(Octree的层级深度), init_level(初始化训练开放的层级深度)
        self.spatial_lr_scale = spatial_lr_scale
        
        # Step1.5: 计算场景包围盒, 确定体素大小(不同于AABB长方体,Octree需要的包围盒是一个正方体,确保能包裹所有SFM点云就行)
        box_min = torch.min(points)*self.extend     # 所有坐标分量的最小值 × 扩展因子,取全局最小值为一个标量
        box_max = torch.max(points)*self.extend     # 所有坐标分量的最大值 × 扩展因子
        box_d = box_max - box_min                   # 所有方向上的最大坐标跨度,而非XYZ三个方向各自跨度
        if self.base_layer < 0:
            # 自动计算基础层: 使最粗体素大小约为 box_d / fork^base_layer
            default_voxel_size = 0.02
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        
        # Octree的本质是把一个大立方体(场景包围盒)递归地切分成8个更小的立方体(fork=2), 直到达到指定层数
        # 初始化从第base_layer层开始,把第base_layer层的体素大小设为voxel_size(最粗的Voxel),之后再逐层细分,每分一层,Voxel边长/2
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)             # 第0层(最粗层)的Voxel边长
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()  # 包围盒原点(立方体的左下后方顶点)
        
        self.octree_sample(points)                  # Step2: 同一批SFM点云经过多层Voxel Grid体素化, 构建八叉树
                                                    # 最粗的Voxel是Octree的第base_layer层,再往下细分K层
        
        # Step3: 可见性裁剪 (weed_out)
        if self.visible_threshold < 0:
            # visible_threshold < 0 表示自动确定阈值: 先以0为阈值裁剪一次, 用平均可见度作为新阈值
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)  # Anchor点可见性裁剪,初始化不同层次的Anchor点可能有冗余,其层级<=相机可见的LOD层级,在渲染时不起作用,会被自动过滤掉
        
        # 打印初始化信息
        logger.info(f'Branches of Tree: {self.fork}')
        logger.info(f'Base Layer of Tree: {self.base_layer}')               # 最粗的Voxel层级是base_layer,Voxel边长是voxel_size
        logger.info(f'Visible Threshold: {self.visible_threshold}')         # Anchor点的可见性阈值,低于此值的锚点被weed_out
        logger.info(f'Appearance Embedding Dimension: {self.appearance_dim}')
        logger.info(f'LOD Levels: {self.levels}')                           # Octree的初始化层级深度K,整个场景用L0-LK-1层anchor
        logger.info(f'Initial Levels: {self.init_level}')                   # 初始激活的LOD层数层(默认K/2)
        logger.info(f'Initial Voxel Number: {self.positions.shape[0]}')     # 初始化Anchor点数量
        logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')  # 最细的Voxel边长(LK-1),由Voxel_Size与K决定
        logger.info(f'Max Voxel Size: {self.voxel_size}')                             # 最粗的Voxel边长(L0),由box_d与base_layer决定

        # Step4: 初始化所有可学习参数
        fused_point_cloud = self.positions
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()    # anchor管辖的Gaussian位置偏移量初始化为0
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()   # anchor特征初始化为0
        
        # 初始化Anchor的缩放参数与旋转参数: 
        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,] KNN中排除自身(idx 0), 取3个邻居,表示每个anchor到其最近的三个anchor邻居的距离平方
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)  # [N, 6] 6维缩放(前3=offset缩放, 后3=高斯缩放),用邻居间距来初始化缩放
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1                                                # 单位四元数 [1, 0, 0, 0] = 无旋转
        # PS：最终高斯位置 = anchor位置 + offset × exp(scales前3维)
        #     最终高斯缩放 = exp(scales后3维) × sigmoid(MLP输出)

        # 将所有参数注册为 nn.Parameter (可学习、可被优化器管理)
        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))     # anchor位置可优化
        self._offset = nn.Parameter(offsets.requires_grad_(True))               # anchor管辖的Gaussian基元偏移量可优化
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))     # anchor特征可优化
        self._scaling = nn.Parameter(scales.requires_grad_(True))               # anchor的缩放可优化: scales 主要作用就是控制 Anchor 周围高斯基元的活动范围和高斯大小(作为MLP_Cov的输入)
        self._rotation = nn.Parameter(rots.requires_grad_(False))               # 旋转不优化(固定)
        self._level = self._level.unsqueeze(dim=1)                              # anchor点所属的Octree层级,[N] → [N, 1] 便于广播
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")  # ΔL初始为0
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")    # Bool变量,每帧动态更新,作用是实现 Octree-GS 的核心LOD选择,选择哪些Anchor点在该相机视角下参与渲染

    # ==========================================================================
    # 【阶段一-Step3】可见性裁剪 — 裁剪初始化Anchor点(在相机视角下,锚点层级小于等于该相机的LOD等级,则该锚点对该相机"可见",否则冗余)
    #                根据相机视角下Anchor点次数统计,根据预设的可见性阈值,移除低频可见的Anchor点
    # ==========================================================================
    def weed_out(self, anchor_positions, anchor_levels):
        """基于LOD可见性裁剪锚点。
        Octree-sample()产生了大量的anchor点,但很多anchor不会被任何训练相机看见(如果一个anchor所处的Level太高,而相机离它太远,那么anchor点永远不会被选中参与渲染==>其实冗余的)
        对每个相机, 计算哪些锚点"应该被看到"(即锚点层级 ≤ 该相机的LOD等级)。
        统计每个锚点在所有相机中的可见频率, 低于阈值的认为是"离群锚点", 被移除。
        
        判断逻辑:
          对每个相机 c:
            1. 计算 L*(c) = log2(d_max / dist(c, anchor)) / log2(fork)
            2. int_level = floor/round/ceil(L*)
            3. 如果 anchor_level ≤ int_level, 则该锚点对该相机"可见" → count += 1
          可见频率 = count / num_cameras
          保留条件: 可见频率 > visible_threshold
        
        参数:
            anchor_positions: [N, 3] 锚点位置坐标(所有Level的anchor点集合)
            anchor_levels:    [N]    锚点对应的层级(所有Level的anchor点层级集合)
        
        返回:
            保留的锚点坐标, 保留的层级, 平均可见度, 保留掩码
        """
        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")  # 为anchor点创建计数器
        for cam in self.cam_infos:  
            # 遍历每一个相机,计算该相机对每个anchor的可见性
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center)**2, dim=1)) * scale  # 计算每个anchor到相机的距离(不是SFM点云的距离)
            
            # LOD公式: L* = log_fork(d_max / dist)  (距离越近, L*越大, 可见层级越多), standard_dist是d_max(相机到SFM点云的最远距离)
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   # 所有anchor点在该相机视角下的预测层级
            int_level = self.map_to_int_level(pred_level, self.levels - 1)          # pred_level是float(如2.7), int_level是int,需要确定pred_level处于哪个层级(默认使用round模式,进行四舍五入)
            # 锚点层级 ≤ 该相机的LOD等级 → 该锚点对该相机可见,统计+1
            visible_count += (anchor_levels <= int_level).int()

        # 遍历完所有相机的视角,统计得到初始化anchor点在所有相机中的可见次数
        visible_count = visible_count/len(self.cam_infos)     # 归一化为[0,1]的可见频率(除以相机数量)
        weed_mask = (visible_count > self.visible_threshold)  # 可见频率低于阈值0.9的anchor点被移除
        mean_visible = torch.mean(visible_count)              # 计算所有anchor点的平均可见频率
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    # ==========================================================================
    # 【阶段二 + 阶段三联动】LOD感知的锚点掩码anchor_mask — 每帧渲染前调用筛选在该相机视角下可见的Anchor点
    #  Anchor的最终可见性受到两方面制约:
    #  1. 距离可见性: 由相机到anchor的距离决定(LOD公式,越近的距离,允许看见Anchor的LOD层级越高)
    #  2. 训练可见性: 由当前迭代步决定(渐进训练,coarse_index决定了当前步数最高可见层级)
    # ==========================================================================
    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        """根据相机到anchor点距离(决定相机所能见的LOD层级)和当前迭代步(用于渐进训练的层级限制), 计算哪些锚点应该参与本帧渲染。
        
        这是 Octree-GS LOD选择的核心函数, 在渲染器 render() 中每帧调用。
        
        LOD公式: L* = log2(d_max / dist) / log2(fork) + ΔL
          - dist 越小(相机越近) → L* 越大 → 允许看见更多精细层Anchor
          - ΔL (extra_level) 提供每锚点的细粒度LOD调整
        
        【阶段三联动 — 渐进训练】:
          如果启用 progressive 模式, coarse_index 会限制当前最高可用层级。
          例如训练初期 coarse_index = init_level = 3, 则只有0~2层的Anchor可见。
          随着训练推进, coarse_index 增加, 逐步解锁更精细的层。
        
        筛选条件: anchor_level ≤ int_level
          即: 只有层级不超过"LOD允许等级"的Anchor才参与渲染。
        
        参数:
            cam_center:       [3] 当前相机的位置
            iteration:        当前训练迭代步数 (用于渐进训练的层级限制)
            resolution_scale: 分辨率缩放因子 (影响距离计算)
        """
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1)) * resolution_scale   # 计算某个位置的相机到每个Anchor的距离(resolution_scale默认1.0不影响距离)
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level   # 计算每个Anchor的预测层级(根据dist距离和fork)
        
        if self.progressive:
            # 渐进训练: 根据当前iteration查找已解锁到第几层
            # searchsorted 查找 iteration 在 coarse_intervals 中的位置
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level  # coarse_index是当前迭代步解锁的LOD最大层级
        else:
            coarse_index = self.levels   # 非渐进模式: 所有层级都可用

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)  # 预测的LOD等级(连续) → 离散层级(上限=coarse_index-1)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)    # 筛选: 锚点层级 ≤ LOD允许等级,Anchor才可见

    # (debug使用)仅仅在函数定义时出现,并没有调用,显示到Cur_level层级,哪些Anchor可见
    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        """与 set_anchor_mask 类似, 但直接指定最高允许层级 cur_level (不依赖iteration)。
        用于致密化阶段按层级单独处理时的掩码计算。
        """
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    # ==========================================================================
    # 优化器初始化与学习率调度
    # ==========================================================================
    def training_setup(self, training_args):
        """初始化Adam优化器、学习率调度器和致密化统计量。
        在训练开始时或从Checkpoint恢复时调用。
        
        为每个参数组分别设置学习率, 并创建对应的指数衰减调度器:
          - anchor:     位置学习率 (乘以 spatial_lr_scale)
          - offset:     偏移学习率 (乘以 spatial_lr_scale)
          - anchor_feat: 特征学习率
          - scaling:    缩放学习率
          - rotation:   旋转学习率
          - mlp_*:      各MLP的学习率
          - embedding:  外观嵌入学习率 (如果使用)
        """
        # 初始化致密化/剪枝统计量 (所有值归零)
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")          # [N, 1],Anchor下所有Gaussian基元的不透明度总和的累积值 ==>用于Anchor剪枝
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")  # [N*k, 1],每个Gaussian基元的屏幕空间(2D)偏移梯度累积值 ==>用于Anchor新增
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")           # [N*k, 1],每个Gaussian基元被渲染的次数                ==>用于Anchor新增
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")           # [N, 1],每个Anchor被观测到的次数(根据相机到anchor距离+训练迭代次数决定LOD上限 ==> 判断Anchor是否可见)
        
        # 学习率参数设置
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        # 学习率调度器参数设置
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        """根据每个训练步更新所有参数组的学习率 (指数衰减)。
        在 train.py 主循环中每步调用: gaussians.update_learning_rate(iteration)
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
    # ==========================================================================
    # PLY点云文件序列化 (save_ply / load_ply)
    # 保存/加载锚点数据用于持久化和后续渲染
    # ==========================================================================
    def construct_list_of_attributes(self):
        """构建PLY文件中各属性列的名称列表。
        顺序: [x,y,z, level, extra_level, f_offset_*, f_anchor_feat_*, scale_*, rot_*]
        每个Anchor共 3+1+1+k*3+feat_dim+6+4 = 77 个float32属性(默认k=10, feat_dim=32)
        """
        l = []
        l.append('x')               # ─┐
        l.append('y')               #  ├─ Anchor的世界坐标 [3], 由Octree体素化确定
        l.append('z')               # ─┘
        l.append('level')           # Anchor所属的Octree层级 (0=最粗, K-1=最细), 决定LOD可见性
        l.append('extra_level')     # ΔL: 每个Anchor的LOD偏移量, 训练中通过anchor_growing逐步增加
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))       # _offset展平: [k,3]→k*3列, Anchor管辖的k个Gaussian基元的位置偏移量
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))  # Anchor特征向量 [feat_dim], 作为MLP输入解码出Gaussian的颜色/不透明度/协方差
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))          # Anchor缩放 [6]: 前3=offset活动范围, 后3=Gaussian大小基底(与MLP_Cov输出相乘)
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))            # 四元数旋转 [4], 固定为[1,0,0,0]不训练, 仅占位
        return l

    def save_ply(self, path, iteration):
        """保存模型到PLY文件。
        只保存当前已解锁层级内的锚点 (渐进训练期间不保存未解锁的层)。
        额外在obj_info中保存 standard_dist 和 levels 元信息。
        """
        mkdir_p(os.path.dirname(path))

        # 若使用渐进式训练,根据当前迭代步数,确定LOD层级已解锁到第几层
        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        # 只保存已解锁层级的锚点
        level_mask = (self._level <= coarse_index-1).squeeze(-1)    # anchor_mask,保存层级小于当前LOD层级上限的anchor
        anchor = self._anchor[level_mask].detach().cpu().numpy()    # anchor点位置
        levels = self._level[level_mask].detach().cpu().numpy()     # anchor层级
        extra_levels = self._extra_level.unsqueeze(dim=1)[level_mask].detach().cpu().numpy()
        anchor_feats = self._anchor_feat[level_mask].detach().cpu().numpy() # anchor点的特征向量
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous()[level_mask].cpu().numpy()
        scales = self._scaling[level_mask].detach().cpu().numpy()
        rots = self._rotation[level_mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 拼接为结构化数组,写入PLY文件 [M,77]
        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, offsets, anchor_feats, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        # 保存额外信息: standard_dist(d_max,计算LOD层级时用) 和 levels(场景初始化时LOD总层数)
        plydata = PlyData([el], obj_info=[
            'standard_dist {:.6f}'.format(self.standard_dist),
            'levels {:.6f}'.format(self.levels),
            ])
        plydata.write(path)

    def load_ply(self, path):
        """从PLY文件加载模型。恢复所有锚点参数和元信息(standard_dist, levels)。
        在渲染/评估时由 Scene 类调用。
        """
        plydata = PlyData.read(path)
        infos = plydata.obj_info
        # 从obj_info恢复元信息 (standard_dist, levels)
        for info in infos:
            var_name = info.split(' ')[0]
            self.__dict__[var_name] = float(info.split(' ')[1])
    
        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int16)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.levels = round(self.levels)
        if self.init_level == -1:
            self.init_level = int(self.levels/2)

    # ==========================================================================
    # 【阶段四】自适应控制 — 锚点剪枝与生长
    # ==========================================================================
    def prune_anchor(self, mask):
        """【阶段四-剪枝】删除被标记的低贡献锚点。
        
        从优化器和所有参数张量中移除 mask=True 的锚点。
        _prune_anchor_optimizer() (继承自BasicModel) 会同时更新Adam优化器的动量状态。
        
        参数:
            mask: [N] bool, True表示要删除的锚点
        """
        valid_points_mask = ~mask    # 取反得到保留掩码

        # 从优化器中剪枝 (同时更新Adam的exp_avg和exp_avg_sq)
        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]        # 层级也要同步裁剪
        self._extra_level = self._extra_level[valid_points_mask]  # ΔL也要同步裁剪

    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask, overlap):
        """【阶段四-生长】基于梯度的多层级锚点生长。Octree-GS 自适应控制的核心。
        
        对每一层 cur_level, 执行三种生长操作:
          (a) 同层生长: 梯度在 [τ_L, τ_{L+1}) 区间 → 在当前LOD层级(L)添加新anchor点
          (b) 跨层生长: 梯度 ≥ τ_{L+1} → 在下一LOD层级(L+1)添加新anchor点
              ⚠ 仅在渐进训练结束后才允许!
          (c) ΔL更新: 锚点级梯度 > τ_L×extra_ratio → _extra_level += extra_up
              ⚠ 仅在渐进训练结束后才执行!
        
        动态阈值公式(LOD层级越高,Voxel越精细,阈值越高):
          τ_L = threshold × (fork^update_ratio)^L
          即更细的层 → 更高的梯度阈值 → 需要更强的证据才能生长
        
        参数:
            iteration:    当前训练迭代步
            grads:        [N*k] 每个神经高斯的平均梯度范数
            threshold:    基础梯度阈值 τ_g (默认0.0002)
            update_ratio: 阈值增长比 β (默认0.2)
            extra_ratio:  ΔL更新的梯度比例 (默认0.25)
            extra_up:     每次ΔL的增量 (默认0.02)
            offset_mask:  [N*k] bool, 观测次数>阈值的神经高斯的mask(确保Gaussian的平均梯度的可靠性)
            overlap:      是否允许重叠生长 (True=允许与已有锚点重叠)
        """
        init_length = self.get_anchor.shape[0]   # 记录初始锚点数 (用于处理循环中新增的锚点)
        grads[~offset_mask] = 0.0                # 统计量不足的神经高斯梯度置0(数据不足,不参与判断)
        # 计算锚点级别的平均梯度 (将Anchor所属的K个Gaussian位置梯度(2D)平均到Anchor上,用于ΔL更新)
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        
        # 遍历LOD每一层级分别处理anchor_growing,有K层的Octree
        for cur_level in range(self.levels):
            # 当前LOD层级的基本信息
            update_value = self.fork ** update_ratio                          # 阈值增长因子: fork^β (默认 2^0.2 ≈ 1.149)
            level_mask = (self.get_level == cur_level).squeeze(dim=1)         # 当前LOD层的Anchor掩码,用于筛选LOD=Cur_Level的Anchor点
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)  # 下一层Cur_Level+1(downscale)的anchor掩码
            if torch.sum(level_mask) == 0:
                continue    # 没有处于LOD=Cur_Level的Anchor点,则Pass
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)  # 当前层体素大小(粗Voxek/2^Cur_Level)
            ds_size = cur_size / self.fork                                # 下一层体素大小
            
            # ===== 计算动态阈值 =====
            # τ_L = τ_g × (fork^β)^L, 更细的层需要更高的梯度才能触发生长
            cur_threshold = threshold * (update_value ** cur_level)   # 当前层的Gaussian梯度阈值 τ_L
            ds_threshold = cur_threshold * update_value               # 下一层的Gaussian梯度阈值 τ_{L+1}
            extra_threshold = cur_threshold * extra_ratio             # ΔL更新阈值
            
            # ===== 基于梯度的三种生长条件(grad是每个高斯基元在2D屏幕上平均梯度) =====
            # (a) 同层生长: τ_L ≤ grad < τ_{L+1}
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)      # candidate_mask(本层LOD阈值)针对Gaussian
            # (b) 跨层生长: grad ≥ τ_{L+1} (梯度很大, 需要更细的层来表示)
            candidate_ds_mask = (grads >= ds_threshold)                             # 针对Gaussian而言,下一层Gaussian阈值
            # (c) ΔL更新: 锚点级梯度 ≥ τ_L × extra_ratio
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            # 处理循环中新增的锚点: 新增锚点不参与生长判断, 用0填充
            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            # 只对当前层的锚点(LOD=cur_level)应用anchor的生成(包括同级/下一级增长)
            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)      # 用于将Anchor级别的Mask扩展到Gaussian级别,用于Anchor所管辖的Gaussian筛选(表明这些GS都是Cur_Level级别Anchor所管辖)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)       # 同层生长候选(筛选同级生长τ_L ≤ grad < τ_{L+1}的Gaussian)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask) # 跨层生长候选(筛选下一级生长 grad ≥ τ_{L+1} 的Gaussian)
            candidate_extra_mask = torch.logical_and(candidate_extra_mask, level_mask)  # ΔL更新候选(筛选本层Anchor, anchor_grads ≥ τ_L × extra_ratio)
            
            # (c) ΔL更新: 仅在渐进训练结束后执行
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()    

            # ===== (a) 同层生长: 在当前层体素网格中添加新锚点 =====
            # 计算候选Gaussian基元的世界坐标 (anchor + offset * scaling)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            # 将当前层(LOD=cur_level)已有Anchor点,转换为网格坐标 (用于去重)
            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size - self.padding).int()
            # 将候选Gaussian基元坐标转换为网格坐标
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]        # 选择的同级生长τ_L ≤ grad < τ_{L+1}的Gaussian(所属LOD=Cur_Level的Anchor管辖)
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size - self.padding).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)   # 进行同级增长的Gaussian对应的网格的去重
            
            # 去重: 去掉与已有锚点重叠的体素 + 可见性检查
            if overlap:
                # overlap模式: 允许重叠, 直接生成候选锚点, 只做可见性weed_out
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask

            elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                # 非重叠模式: 先去除与已有锚点重复的体素, 再做weed_out
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates  # 取反: 保留“不重复”的
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
                
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            # ===== (b) 跨层生长: 在下一层(L+1)体素网格中添加新锚点 =====
            # 同样的去重 + weed_out 流程, 但用更细的ds_size体素网格
            grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size-self.padding).int()
            selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
            selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size-self.padding).int()
            selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
            # 跨层生长仅在渐进训练结束后且未到最细层时才执行
            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                if overlap:
                    remove_duplicates_ds =  torch.ones(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device="cuda")
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                elif selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            # ===== 将同层+跨层的新锚点合并, 初始化属性并加入优化器 =====
            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()
                
                # 特征初始化: 从父锚点继承特征, 同一体素内多个候选取max聚合
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_ds_mask]
                new_feat_ds = scatter_max(new_feat_ds, inverse_indices_ds.unsqueeze(1).expand(-1, new_feat_ds.size(1)), dim=0)[0][remove_duplicates_ds]
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)
                
                # 缩放初始化: 同层用cur_size, 跨层用ds_size (log空间, 因为用exp激活)
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1,2]).float().cuda()*ds_size
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)
                
                # 旋转初始化为单位四元数
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                # 偏移量初始化为0, ΔL初始化为0
                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])
                
                d = {
                    "anchor": new_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                }   

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)
    
    # ==========================================================================
    # 【阶段四-编排】完整的致密化流程: 生长 + 剪枝,每100步调用一次run_densify()
    # ==========================================================================
    def run_densify(self, iteration, opt):
        """【阶段四】执行一次完整的致密化操作: 生长新锚点 + 剪枝低贡献锚点。
        在 train.py 中每隔 update_interval(默认100)步调用一次。
        
        流程:
          1. 计算Gaussian基元平均梯度 = offset_gradient_accum / offset_denom
          2. anchor_growing() → 基于梯度生长新Anchor
          3. 重置已使用的梯度统计量, 并为新增Anchor补充统计空间
          4. 剪枝: 不透明度累积 < min_opacity × 渲染次数 的锚点被删除
          5. 重置剪枝后的统计量
        
        剪枝条件 (两个条件同时满足才剪):
          - 不透明度低: opacity_accum < min_opacity × anchor_demon
          - 被足够多次渲染: anchor_demon > update_interval × success_threshold
          即: 被足够多次渲染但平均不透明度仍然很低的锚点会被删除。
        
        参数:
            iteration: 当前训练迭代步
            opt:       优化参数, 包含 densify_grad_threshold, update_ratio, 
                       extra_ratio, extra_up, min_opacity, success_threshold 等
        """
        # ===== Step 1: 计算平均梯度 =====
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1],每个Gaussian基元的累积2D梯度/被渲染次数
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1) # [N*k, 1],取范数,Gaussian基元平均梯度的模长
        offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)    # 观测次数超过阈值的Gaussian的累积梯度才有可靠性
        
        # ===== Step 2: 锚点生长 =====
        self.anchor_growing(iteration, grads_norm, opt.densify_grad_threshold, opt.update_ratio, opt.extra_ratio, opt.extra_up, offset_mask, opt.overlap)
        
        # ===== Step 3: 重置梯度统计量并为新增锚点补充空间 =====
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # ===== Step 4: 锚点剪枝 =====
        # 剪枝条件1: 不透明度低于阈值 (opacity_accum < min_opacity × 渲染次数)
        prune_mask = (self.opacity_accum < opt.min_opacity*self.anchor_demon).squeeze(dim=1)
        # 剪枝条件2: 被足够多次渲染 (避免误剪刚初始化的锚点)
        anchors_mask = (self.anchor_demon > opt.update_interval * opt.success_threshold).squeeze(dim=1)
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # 两个条件同时满足才剪
        
        # ===== Step 5: 更新剪枝后的统计量 =====
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # 重置被足够多次渲染的锚点的统计量 (为下一轮致密化重新累积)
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

    # ==========================================================================
    # MLP Checkpoint保存/加载 (TorchScript格式, 用于部署)
    # ==========================================================================
    def save_mlp_checkpoints(self, path):
        """将MLP网络编译为TorchScript格式保存。用于无Python环境的推理部署。"""
        mkdir_p(os.path.dirname(path))
        self.eval()
        opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
        cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
        color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.appearance_dim).cuda()))
        color_mlp.save(os.path.join(path, 'color_mlp.pt'))
        if self.use_feat_bank:
            feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, self.view_dim).cuda()))
            feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
        if self.appearance_dim > 0:
            emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
            emd.save(os.path.join(path, 'embedding_appearance.pt'))
        self.train()

    def load_mlp_checkpoints(self, path):
        """从TorchScript格式加载MLP网络。"""
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        if self.use_feat_bank:
            self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
        if self.appearance_dim > 0:
            self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
    
    # ==========================================================================
    # 【阶段二/五】核心前向函数: 将锚点特征解码为神经高斯属性
    # 在渲染器 render() 中调用, 是 Scaffold-GS/Octree-GS 的核心解码流程
    # ==========================================================================
    def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, ape_code=-1):
        """【阶段二/五】从锚点特征解码生成神经高斯的完整属性。
        
        这是 Scaffold-GS/Octree-GS 的核心前向函数, 在渲染器 render() 中调用。
        将锚点的特征向量通过MLP解码为每个神经高斯的: 位置、颜色、不透明度、缩放、旋转。
        
        解码流程:
          1. 提取可见锚点的特征和属性
          2. 计算观察方向 (anchor → camera 的单位向量)
          3. (可选) 特征银行: 多分辨率特征加权融合
          4. 拼接 [特征 + 观察方向] 作为MLP输入
          5. 【阶段五】(可选) 拼接外观嵌入码
          6. MLP解码: opacity(不透明度), color(颜色), cov(协方差=缩放+旋转)
          7. 【progressive模式】对临界层的不透明度做平滑过渡
          8. 后处理: 计算最终的高斯位置 = anchor + offset * scaling
        
        参数:
            viewpoint_camera: 当前渲染视角的相机对象
            visible_mask:     [N] bool, 可见锚点掩码 (来自 prefilter_voxel)
            ape_code:         外观嵌入码索引 (-1=使用相机uid)
        
        返回:
            xyz:     [M, 3]   最终高斯中心坐标 (anchor + offset * scaling)
            color:   [M, 3]   高斯颜色 RGB
            opacity: [M, 1]   高斯不透明度
            scaling: [M, 3]   高斯缩放
            rot:     [M, 4]   高斯旋转四元数
            None:    预留位 (原版中无用)
            mask:    [N*k]    有效高斯的掩码 (opacity > 0)
        """
        # ===== Step 1: 提取可见锚点的数据 =====
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]          # [n, 3] 可见锚点坐标
        feat = self.get_anchor_feat[visible_mask]       # [n, c] 可见锚点特征
        grid_offsets = self.get_offset[visible_mask]     # [n, k, 3] 可见锚点的偏移量
        grid_scaling = self.get_scaling[visible_mask]    # [n, 6] 可见锚点的缩放参数

        # ===== Step 2: 计算观察方向 =====
        ob_view = anchor - viewpoint_camera.camera_center   # [n, 3] 锚点→相机的方向向量
        ob_dist = ob_view.norm(dim=1, keepdim=True)         # [n, 1] 锚点到相机的距离
        ob_view = ob_view / ob_dist                         # [n, 3] 归一化为单位方向向量

        # ===== Step 3: (可选) 特征银行: 视角自适应的多分辨率特征融合 =====
        if self.use_feat_bank:
            bank_weight = self.get_featurebank_mlp(ob_view).unsqueeze(dim=1) # [n, 1, 3] 三个分辨率的权重

            # 将特征分为三个频率: 1/4采样(低频) + 1/2采样(中频) + 全采样(高频)
            # 用视角依赖的权重加权融合
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                feat[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1) # [n, c]

        # ===== Step 4: 拼接MLP输入 = [锚点特征 + 观察方向] =====
        cat_local_view = torch.cat([feat, ob_view], dim=1) # [N, feat_dim+view_dim]

        # ===== Step 5: 【阶段五】(可选) 获取外观嵌入码 =====
        if self.appearance_dim > 0:
            if ape_code < 0:
                # 训练时: 用相机的uid索引外观嵌入表
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
                appearance = self.get_appearance(camera_indicies)
            else:
                # 指定外观码 (用于跨相机外观迁移等)
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
                appearance = self.get_appearance(camera_indicies)
                
        # ===== Step 6: MLP解码 =====
        # 6a. 不透明度: [N, k] 每个锚点的k个高斯的不透明度
        neural_opacity = self.get_opacity_mlp(cat_local_view) # [N, k]

        # 【progressive模式】对处于临界层的锚点做不透明度平滑过渡
        # prog_ratio 是小数部分, 用于在层级边界处做软过渡, 避免突变
        if self.dist2level=="progressive":
            prog = self._prog_ratio[visible_mask]
            transition_mask = self.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0    # 非临界层的不透明度保持不变
            neural_opacity = neural_opacity * prog  # 临界层的不透明度乘以过渡系数
        
        # 不透明度掩码: 只保留 opacity > 0 的高斯 (Tanh输出负值的被剔除)
        neural_opacity = neural_opacity.reshape([-1, 1])   # [N*k, 1]
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)            # [N*k] 有效高斯的掩码

        opacity = neural_opacity[mask]  # [M, 1] 保留的有效不透明度

        # 6b. 颜色: [N, k*3] → [N*k, 3]
        # 【阶段五】如果使用外观嵌入, 输入 = [特征+观察方向+外观码]
        if self.appearance_dim > 0:
            color = self.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = self.get_color_mlp(cat_local_view)
        color = color.reshape([anchor.shape[0]*self.n_offsets, 3])

        # 6c. 协方差: [N, k*7] → [N*k, 7] (3维缩放 + 4维旋转)
        scale_rot = self.get_cov_mlp(cat_local_view)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7])
        
        # 偏移量展开: [N, k, 3] → [N*k, 3]
        offsets = grid_offsets.view([-1, 3])
        
        # ===== Step 7: 并行掩码 + 后处理 =====
        # 将所有属性拼接后统一应用mask, 提高GPU并行效率
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N, 9] = [6+3]
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)  # [N*k, 9]
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)  # [N*k, 22]
        masked = concatenated_all[mask]     # [M, 22] 只保留有效高斯
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
        
        # 后处理协方差: 
        # 最终缩放 = 锚点缩放(后3维) × sigmoid(MLP输出前3维)
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
        # 旋转: normalize确保四元数为单位长度
        rot = self.rotation_activation(scale_rot[:,3:7])
        
        # ===== Step 8: 计算最终高斯中心坐标 =====
        # xyz = anchor_position + offset * anchor_scaling(前3维)
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets 
        
        return xyz, color, opacity, scaling, rot, None, mask