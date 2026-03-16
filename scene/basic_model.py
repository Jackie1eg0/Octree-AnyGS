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
basic_model.py — Scaffold-GS / Octree-GS 高斯模型基类

本文件定义了 BasicModel 基类, 作为所有高斯模型 (GaussianModel, GaussianLoDModel 等) 的父类。
它提供了模型训练和致密化过程中的通用功能:

# 用 LOD 配置时，主要看 lod_model.py + basic_model.py
┌──────────────────────────────────────────────────────────────────┐
│                        BasicModel (基类)                          │
├──────────────────────────────────────────────────────────────────┤
│ 1. 激活函数设置: setup_functions()                                │
│ 2. 优化器管理:                                                    │
│    - replace_tensor_to_optimizer(): 替换优化器中的参数             │
│    - cat_tensors_to_optimizer():    向优化器追加新参数(锚点生长)    │
│    - _prune_anchor_optimizer():     从优化器移除参数(锚点剪枝)      │
│ 3. 训练统计: training_statis()  收集梯度/不透明度统计量             │
│ 4. 去重工具: get_remove_duplicates()  体素网格坐标去重              │
│ 5. LOD映射: map_to_int_level()  连续LOD→离散层级映射               │
│ 6. 默认空实现: eval/train/set_appearance 等(子类可覆盖)            │
└──────────────────────────────────────────────────────────────────┘

子类关系:
  BasicModel (本文件)
    ├── GaussianModel      (gs_model_scaffoldgs/base_model.py)  — 原版Scaffold-GS
    └── GaussianLoDModel   (gs_model_scaffoldgs/lod_model.py) — Octree-GS (LOD版本)
"""

import os
import torch
from torch import nn
from functools import reduce
from utils.general_utils import inverse_sigmoid
    
class BasicModel:
    """Scaffold-GS / Octree-GS 高斯模型基类。
    
    不直接实例化, 而是被 GaussianModel 或 GaussianLoDModel 继承。
    提供优化器管理、训练统计、LOD映射等通用功能。
    """

    # ==========================================================================
    # 激活函数设置
    # ==========================================================================
    def setup_functions(self):
        """设置各属性的激活函数和反激活函数。
        
        高斯的参数在内部以"原始值"存储, 需要经过激活函数变换到"物理值":
          - scaling:  原始值 → exp() → 正数缩放值
          - opacity:  原始值 → sigmoid() → [0,1] 不透明度
          - rotation: 原始值 → normalize() → 单位四元数
        
        反激活函数用于将"物理值"转回"原始值"(如初始化时使用)。
        """
        self.scaling_activation = torch.exp               # 缩放: exp确保正数
        self.scaling_inverse_activation = torch.log       # 缩放反函数: log
        self.opacity_activation = torch.sigmoid           # 不透明度: sigmoid→[0,1]
        self.inverse_opacity_activation = inverse_sigmoid # 不透明度反函数
        self.rotation_activation = torch.nn.functional.normalize  # 旋转: 归一化为单位四元数
    
    # ==========================================================================
    # 默认空实现 (子类可覆盖)
    # 以下方法在BasicModel中提供空实现或最简实现,
    # 子类(如GaussianLoDModel)会覆盖这些方法添加实际功能。
    # ==========================================================================
    def eval(self):
        """切换到评估模式。基类空实现, GaussianLoDModel 会覆盖为切换MLP到eval模式。"""
        return

    def train(self):
        """切换到训练模式。基类空实现, GaussianLoDModel 会覆盖为切换MLP到train模式。"""
        return

    def set_appearance(self, num_cameras):
        """初始化外观嵌入。基类默认不使用外观嵌入 (设为None)。
        GaussianLoDModel 会覆盖为创建 Embedding(num_cameras, appearance_dim)。
        """
        self.embedding_appearance = None
    
    def oneupSHdegree(self):
        """球谐函数(SH)阶数递增。用于颜色表示的渐进训练。
        在 Scaffold-GS 中通常不使用 (用MLP代替SH), 但保留接口兼容性。
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_coarse_interval(self, opt):
        """设置渐进训练的层级解锁时间表。基类空实现。
        GaussianLoDModel 会覆盖为计算等比数列解锁时间表。
        """
        return 
        
    def set_anchor_mask(self, *args):
        """设置锚点可见性掩码。基类默认所有锚点都可见。
        GaussianLoDModel 会覆盖为基于LOD公式计算距离自适应的掩码。
        """
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
    
    # ==========================================================================
    # 优化器管理 — Adam优化器的参数动态增删
    # Scaffold-GS/Octree-GS 在训练过程中会动态增减锚点,
    # 必须同步更新Adam优化器中的动量(exp_avg)和二阶矩(exp_avg_sq)
    # ==========================================================================
    def replace_tensor_to_optimizer(self, tensor, name):
        """替换优化器中指定参数组的张量, 并重置其Adam动量。
        
        用途: 当某个参数需要完全重新初始化时 (如重置缩放参数),
        需要同时替换优化器中的张量和对应的Adam状态。
        
        实现逻辑:
          1. 在optimizer.param_groups中找到name匹配的参数组
          2. 将Adam的 exp_avg, exp_avg_sq 重置为全零 (丢弃历史动量)
          3. 用新tensor替换旧参数
          4. 更新优化器状态字典的键映射
        
        参数:
            tensor: 新的参数张量
            name:   参数组名称 (如 "anchor", "scaling" 等)
        
        返回:
            dict, {name: nn.Parameter} 更新后的参数字典
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # 重置Adam动量为零 (新参数没有历史梯度信息)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 更新优化器状态: 删除旧键 → 设置新参数 → 关联新键
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        """向优化器中的参数追加新张量。在锚点生长(anchor_growing)时调用。
        
        当新锚点被创建后, 需要将它们的参数拼接到优化器现有参数的末尾,
        同时为新参数创建零初始化的Adam动量状态。
        
        注意: 跳过MLP、卷积和嵌入层的参数 (它们不随锚点数量变化)。
        
        实现逻辑:
          1. 遍历所有参数组, 跳过mlp/conv/embedding
          2. 获取该参数组的扩展张量 (如 new_anchor, new_scaling 等)
          3. 将 exp_avg, exp_avg_sq 也拼接上零向量
          4. 参数: old_tensor cat new_tensor = extended_tensor
        
        参数:
            tensors_dict: dict, {param_name: extension_tensor}
                          例如 {"anchor": new_anchor, "scaling": new_scaling, ...}
        
        返回:
            dict, {name: nn.Parameter} 拼接后的完整参数字典
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # MLP/卷积/嵌入层参数大小固定, 不随锚点增减, 跳过
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]  # 要追加的新参数
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 有Adam状态: 同时拼接动量和二阶矩 (新参数部分用0填充)
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 无Adam状态 (首次使用): 直接拼接
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # ==================== (核心模块:每帧渲染后调用,指导Anchor的grow & prune) ======================================
    # 训练统计收集 — 为致密化(阶段四)提供决策依据
    # 每个训练步的渲染结束后调用, 累积梯度和不透明度统计
    # ==========================================================================
    def training_statis(self, render_pkg, width, height):
        """收集每帧渲染的梯度和不透明度统计信息, 用于指导致密化决策。
        
        在 train.py 的主循环中, 每次渲染后调用。收集三类统计量:
          1. opacity_accum:         每个锚点的不透明度累积 (用于剪枝判断)
          2. anchor_demon:          每个锚点被渲染的次数 (用于归一化)
          3. offset_gradient_accum: 每个神经高斯的屏幕空间2D梯度累积 (用于生长判断)
        
        实现逻辑:
          Step 1: 从 render_pkg 中提取渲染结果
          Step 2: 累积每个Anchor所管辖的所有Gaussians的不透明度总和 → opacity_accum
          Step 3: 可见锚点的渲染计数 +1 → anchor_demon
          Step 4: 计算屏幕空间梯度范数, 累积到 offset_gradient_accum
        
        参数:
            render_pkg: dict, 渲染器返回的字典, 包含一帧渲染的所有输出结果:
                render:            [3, H, W] 渲染出来的RGB图像
                scaling:           [M, 3] 每个Gaussian的缩放参数(假设Gaussian基元个数为M个)
                viewspace_points:  [1, M, 2] 每个Gaussian在屏幕空间的2D坐标(means2d),带梯度,是致密化统计的核心
                visibility_filter: [M] bool, 在该帧图片中实际被光栅化渲染的Gaussian Mask(真正对最终像素有贡献的Gaussian基元)
                visible_mask:      [N] bool, Anchor级别:可见锚点掩码两层筛选 (LOD等级筛选仅仅保留level<=floor(L*)的Anchor+来自prefilter_voxel视锥剔除)
                selection_mask:    [N*k] bool, Gaussian级别:有效的高斯基元掩码 (opacity>0,MLP预测GS属性的时候,可以通过Opacity<=0来动态调整Anchor周围实际管辖的有效GS个数)
                opacity:           [M, 1] Gaussian级别:每个Gaussian基元的不透明度值(Opacity)
            width:  渲染图像宽度 (用于梯度缩放)
            height: 渲染图像高度 (用于梯度缩放)
        
        # 按照gaussian_renderer/render.py中的Mask调用顺序,理解如何从N个Anchor管辖的N*K个Gaussian ----(一步步筛选)----> 该帧(Image)渲染所涉及的Gaussian:
        #   N 个锚点 (全部Anchor) --> 总共具有N*K个Gaussian(有效GS+无效GS) PS:此处有效+无效指的是MLP预测的GS Opacity>0还是<=0
        #   
        #   Step1(Anchor点LOD级别筛选): 第一层筛选(_anchor_mask LOD层级筛选): 该帧Image具有相机的位姿(包含相机坐标+相机朝向),Anchor点的位置可知(不同LOD层级的Anchor点位置可确定) ==> 可用Anchor到相机中心距离计算LOD层级L*,将L*映射为离散整数(round/progressive)
        #          筛选条件: anchor_level <= int_level 只要LOD层级<=该相机可见的最大LOD层级,就留下,因为相机可观察到这些层级
        #          从每个Anchor点出发(每个Anchor在增加的时候其LOD层级是确定的,位置确定的),计算每个Anchor点到所有相机中心的欧式距离,全量一次性计算,再确定该相机所见的最大LOD层级,满足anchor_level<=int_level的Anchor才会被保留
        #          
        #   Step2(Anchor点视锥剔除): 第二层筛选(visible_mask Anchor级别的视锥剔除Frustum Culling ), 经过第一层剔出的Gaussian仅满足其所属Anchor的LOD层级<=该相机到Anchor最大可见层级,但是Anchor可能不在相机视锥以内
        #          视锥之外的Anchor对该帧的渲染起不到任何作用,其经过Splat不会在屏幕留下痕迹,因此可以做视锥剔除
        #   疑难区分:视锥体的剔除逻辑:是Anchor(有Scaling前三维参数,可以视为一个粗粒度的3DGS)经过Splat到2D屏幕看是否有投影的2D Gaussian Or 从屏幕出发根据相机内存+外参构建3D视锥体,看Anchor点是否在视锥体内
        #   答案是第一种:Anchor在做视锥剔除时,前3维 scaling + rotation 定义了一个 3D 椭球体,按照Splat到2D屏幕上,若Radii>0(2D 椭球在平面上显现),其管辖的Gaussian才有资格进入后续的MLP解码
        #       得到的Visible_mask[N],Bool型Mask,同时编码了LOD筛选+视锥剔除 N ==> N_Vis
        #   
        #   Step3(Gaussian级别剔除,MLP预测有效的Gaussian满足Opacity>0):为减小MLP计算开销,只有经过LOD级别筛选+视锥剔除得到的N_Vis数量的Anchor点,才能进入MLP预测
        #           提取N_Vis个筛选的Anchor点(LOD级别满足要求+落在该帧相机的视锥以内),N_Vis个Anchor点特征,每个Anchor点特征32维度+View_dim(相机到Anchor观察方向)==> 作为MLP_Opacity输入
        #           N_Vis个Anchor点特征+View_dim(相机到Anchor观察方向)==> MLP_Opacity(N_Vis,n_offsets) ==> 得到selection_mask[N_Vis*n_offsets],True的数量是有效的GS数量(Opacity>0)
        #
        #   Step4(Gaussian级别筛选,光栅化投影Gaussian,筛选出真正参与了像素着色的Gaussian)
        #           虽然Gaussian经过上述三步骤筛选,visibility_filter是光栅化过程的自然产品,需要注意两点
        #           1.Anchor级别视锥剔除≠Gaussian级别视锥剔除:Step2的视锥剔除是在Anchor级别做的,但是不代表其管辖的Gaussian都在视锥以内,因为GS的最终位置是anchor+offset*scaling[:3],因此一个Anchor在屏幕中央,但其管辖的GS可能通过Offset偏移到屏幕之外,投影后radii=0,被剔除
        #           2.即使Anchor所属的Gaussian在视锥以内,radii也可能为0,一方面可能Gaussian的Scaling极小,投影到屏幕椭圆半径接近0 另一方面GS位于近平面之后被裁剪  
        #   光栅化的过程: 3D Gaussian Splatting 到 2D 屏幕(涉及坐标系转换:W2C + 3D Gaussian --Splat-->2D Gaussian(屏幕空间))
        #                2D 协方差 → 投影半径 radii(对2*2的Σ2D求最大特征值,取3σ范围的整数Pixel半径)
        #                屏幕范围检查: GS中心投影到屏幕像素坐标(Px,Py),检查[px-radii, px+radii] X [py-radii, py+radii]是否与屏幕[0,W] X [0,H]重叠 完全不相交 ==> radii=0
        #                radii=0的GS有如下情况: 1)深度在近/远平面之外 2)Scaling极小 3)投影的Gaussian完全在屏幕之外
        #   着色阶段:Tile-Based Rasterization,将屏幕分成多个Tile,深度排序,逐像素着色,每个Tile独立着色,可以并行处理
        #   
        #   Step5(Anchor累积Opacity):Step1+2+3筛选后的Gaussian(不包含Step4光栅化剔除的Gaussian):为判断Anchor是否需要剪枝,即使GS没有被光栅化渲染(radii=0),但只要MLP认为其有效Opacity>0,就说明这个Anchor还在做渲染贡献
        #        (Anchor梯度累积):用的是Step1+2+3+4筛选后的Gaussian,判断Anchor是否需要增长,只有真正参与Pixel着色的Gaussian才有有效2D屏幕梯度,必须经过Step4光栅化
        
        
        # Q:渲染损失Loss反向传播 ==> 2D Gaussian位置梯度
        # L1_loss计算Render出来的像素与真实拍摄的像素差异,渲染出的Pixel颜色是根据α-Blending公式计算的,GS的颜色*GS的不透明度*累积的透光率
        #                  Loss → C(p) → αᵢ → G(p, μᵢ, Σᵢ) → μᵢ²ᴰ ✅ 
        #                  而Render出来的像素颜色C_p在α-Blending公式中受到GS的Opacity的影响
        #                  Opacity是根据该Pixel到2D Gaussian的距离决定的(GS本身不透明度*2D高斯概率)
        """
        # =====================================================================
        # Step 1: 从 render_pkg 字典中提取本帧渲染的关键结果
        # =====================================================================
        # render_pkg 由 gaussian_renderer/render.py 的 render() 或 render_2dgs() 返回
        # 其中 N=锚点总数, k=n_offsets(每锚点生成的高斯数), M=实际有效的神经高斯数
        
        viewspace_point_tensor = render_pkg["viewspace_points"]
        # viewspace_points: [1, M, 2] — 每个神经高斯在屏幕空间的2D坐标(means2d)
        #   这个张量在渲染时调用了 retain_grad(), 因此反向传播后 .grad 存储了
        #   损失函数对屏幕空间坐标的梯度。梯度大 → 该位置渲染误差通过该高斯传播强 → 说明该区域欠拟合, 需要更多锚点

        update_filter = render_pkg["visibility_filter"]
        # visibility_filter: [M] bool — 光栅化时 radii > 0 的高斯(真正参与了像素着色的Gaussian基元)
        #   即实际参与了像素着色的高斯。有些高斯虽然通过了 prefilter(视锥剔除+LOD筛选),但在光栅化阶段可能因投影半径为0而被跳过

        anchor_visible_mask = render_pkg["visible_mask"]
        # visible_mask: [N] bool — prefilter_voxel() 阶段确定的可见锚点
        #   若为True = 该锚点通过了 LOD筛选 + 视锥剔除, 参与了本帧渲染

        offset_selection_mask = render_pkg["selection_mask"]
        # selection_mask: [N*k] bool — generate_neural_gaussians() 中标记的有效偏移
        #   每个锚点生成 k 个候选高斯, 但只有 opacity > 0 的才是"有效高斯"(MLP预测GS属性的时候,可以通过Opacity<=0来动态调整Anchor周围实际管辖的有效GS个数)
        #   True的数量 = M (有效神经高斯数)

        opacity = render_pkg["opacity"]
        # opacity: [M, 1] — 每个高斯基元的不透明度值 (经sigmoid激活后)
        
        # =====================================================================
        # Step 2: 累积每个锚点的不透明度总和(Step1+Step2+Step3筛选) → self.opacity_accum
        # =====================================================================
        # 用途: prune_anchor() 中, 如果 opacity_accum / anchor_demon < min_opacity,
        #       说明该锚点长期贡献很低, 会被剪枝删除
        #
        # 思路: opacity 只包含 M 个有效高斯的值(MLP预测GS属性若Opacity<=0,则该位置的GS不会被渲染,仅仅占位), 但我们需要按锚点(N_Anchor)汇总,
        #       所以先创建 [N_Anchor*k] 的全零数组, 把 M 个值填回对应位置, 再 reshape 求和

        # 创建全零数组, 大小 = Anchor所管辖的全部高斯数(有效+无效) [N*k]
        temp_opacity = torch.zeros(offset_selection_mask.shape[0], dtype=torch.float32, device="cuda")
        # 将有效高斯(MLP预测出的不透明度>0的GS)的opacity填入对应位置 (offset_selection_mask为True的位置)
        # clone().detach(): 不参与梯度计算, 仅作统计用
        temp_opacity[offset_selection_mask] = opacity.clone().view(-1).detach()
        
        # reshape 为 [N, k], 每行是一个锚点的 k 个偏移位置的 opacity
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        # 按锚点求和(dim=1), 累加到 opacity_accum 中对应的可见锚点(只有Opacity>0的有效GS才会被渲染,才会把自己的不透明度加到对应的锚点上)
        # keepdim=True 保持 [N_visible, 1] 的形状, 与 opacity_accum 对齐
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # =====================================================================
        # Step 3: 累积锚点被渲染的次数 → self.anchor_demon
        # =====================================================================
        # 用途: 作为 opacity_accum 的归一化分母, 平均opacity = opacity_accum / anchor_demon
        #       只有被渲染足够多次(> update_interval * success_threshold)
        #       且平均opacity仍然很低的锚点, 才会被剪枝
        self.anchor_demon[anchor_visible_mask] += 1 # 在该帧可见的Anchor点统计次数+1

        # =====================================================================
        # Step 4: Anchor累积2D GS屏幕空间梯度范数 → self.offset_gradient_accum
        # =====================================================================
        # 用途: anchor_growing() 中, 如果某个高斯的平均梯度 > 阈值 τ_L,
        #       说明该位置欠拟合, 需要在相应层级新增锚点
        #
        # 难点: 梯度 grad 的索引空间是 [M] (有效且被光栅化渲染的高斯),
        #       但 offset_gradient_accum 的索引空间是 [N*k] (所有偏移位置),
        #       因此需要构建 combined_mask 来做映射:
        #       combined_mask[i] = True 表示全局偏移位置 i 对应一个 "既是有效高斯, 又实际被渲染" 的高斯基元

        # --- 构建 combined_mask: 两步筛选 ---
        # Step 4a: 将 anchor_visible_mask 从 [N] 展开为 [N*k]
        #   每个锚点所管辖的 k 个Gaussian基元都继承该锚点的可见性
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)

        # Step 4b: 初始化全零的 combined_mask [N*k]
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)

        # Step 4c: 第一步筛选 — 标记"有效高斯"(MLP预测的Opacity>0的GS)
        #   在可见锚点对应的偏移位置中, 标记 offset_selection_mask(MLP预测Opacity>0的GS基元Mask)为True的位置
        #   anchor_visible_mask: [N*k] bool,该Gaussian对应的Anchor是否可见
        #   offset_selection_mask: [N*k] bool,该Gaussian是否有效(MLP预测的Opacity>0)  ===> 只有Anchor点可见+Opacity>0的Gaussian才会被累积梯度
        combined_mask[anchor_visible_mask] = offset_selection_mask

        # Step 4d: 第二步筛选 — 从有效高斯中再筛选"实际被光栅化渲染"的GS
        #   虽然Gaussian对应的Anchor点有效+Opacity>0,但Gaussian可能对该帧图像渲染的贡献很小(radii=0)或者不参与渲染,需筛选
        #   update_filter[i]=False的高斯虽然有效但没被光栅化(radii=0), 排除掉
        #   最终 combined_mask中True的数量 = 实际被渲染的高斯数M'(update_filter为True的数量)
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        # --- 计算梯度范数并累积 ---
        # viewspace_point_tensor.grad: [1, M, 2] — 损失对2D屏幕空间坐标的梯度
        # 这个梯度是 loss.backward() 自动计算的, 因为 means2d 调用了 retain_grad()
        grad = viewspace_point_tensor.grad.squeeze(0)  # [M, 2]

        # NDC坐标 → 像素坐标的缩放
        # NDC范围大约 [-1, 1], 乘以 width/2 和 height/2 转换为像素单位
        # 这样不同分辨率的图像产生的梯度具有可比性
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5

        # 只取实际被光栅化渲染的GS的梯度, 计算L2范数
        # grad_norm: [M', 1] — 每个被渲染高斯的屏幕空间梯度范数
        grad_norm = torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)

        # 通过 combined_mask 将梯度范数累积到全局偏移位置数组中
        # 后续 anchor_growing() 会用 offset_gradient_accum / offset_denom
        # 计算每个偏移位置的平均梯度, 与阈值 τ_L 比较来决定是否生长
        self.offset_gradient_accum[combined_mask] += grad_norm  # 记录每个Gaussian在过去多少帧中的位置梯度总和(只有在光栅化渲染时候才进行该GS位置梯度累加)
        self.offset_denom[combined_mask] += 1                   # 记录GS在多少帧中被光栅化渲染(GS被光栅化渲染的次数)
        
    # ==========================================================================
    # 优化器剪枝 — 从Adam优化器中移除被删除的锚点
    # ==========================================================================
    def _prune_anchor_optimizer(self, mask):
        """从优化器中移除被mask标记为False的锚点, 同步更新Adam状态。
        
        在 prune_anchor() 中调用。与 cat_tensors_to_optimizer() 相反,
        这里是按mask选择保留的参数行, 缩小参数张量。
        
        特殊处理: 对"scaling"参数的后3维(高斯缩放)做截断,
        确保缩放不超过0.05 (防止剪枝后残留过大的高斯)。
        
        实现逻辑:
          1. 遍历所有参数组, 跳过mlp/conv/embedding
          2. 按mask索引 exp_avg 和 exp_avg_sq (保留对应行)
          3. 按mask索引参数本身
          4. 对scaling后3维做 min(值, 0.05) 的截断
        
        参数:
            mask: [N] bool, True=保留, False=删除
        
        返回:
            dict, {name: nn.Parameter} 剪枝后的参数字典
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 跳过MLP/卷积/嵌入层 (大小不随锚点变化)
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 按mask保留Adam动量的对应行
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                # 特殊处理: 截断高斯缩放 (scaling后3维), 防止剪枝后残留过大的高斯
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]          # 后3维 = 高斯缩放 (前3维 = offset缩放)
                    temp[temp>0.05] = 0.05       # 上限截断
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    # def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):

    #     if use_chunk:
    #         chunk_size = 4096
    #         max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
    #         remove_duplicates_list = []
    #         for i in range(max_iters):
    #             cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
    #             remove_duplicates_list.append(cur_remove_duplicates)
    #         remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
    #     else:
    #         remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
    #     return remove_duplicates

    # ==========================================================================
    # 体素网格去重 — 生长时检测候选位置是否与已有锚点重叠
    # ==========================================================================
    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, num_overlap=1, use_chunk=True):
        """检测候选体素坐标是否与已有体素坐标重叠。在 anchor_growing() 中调用。
        
        对每个候选体素, 统计它与已有体素的匹配次数。
        匹配次数 >= num_overlap 的候选体素被认为是"重复的", 应该被排除。
        
        实现逻辑:
          对每个候选坐标, 逐元素比较是否与某个已有坐标完全相同 (.all(-1))。
          统计匹配数量, 超过 num_overlap 阈值则标记为重复。
          
          use_chunk=True 时分块处理, 避免 O(M*N) 的显存爆炸。
        
        参数:
            grid_coords:                    [M, 3] int, 已有锚点的体素网格坐标
            selected_grid_coords_unique:    [C, 3] int, 候选新锚点的唯一体素坐标
            num_overlap:                    重叠阈值 (默认1), >= 该值视为重复
            use_chunk:                      是否分块处理 (默认True, 节省显存)
        
        返回:
            [C] bool, True=该候选坐标与已有坐标重复
        """
        counts = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.int, device=selected_grid_coords_unique.device)

        if use_chunk:
            chunk_size = 4096  # 每次处理4096个已有坐标, 避免显存溢出
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            for i in range(max_iters):
                chunk = grid_coords[i * chunk_size:(i + 1) * chunk_size]
                # [C, 1, 3] == [1, chunk, 3] → [C, chunk, 3] → all(-1) → [C, chunk]
                matches = (selected_grid_coords_unique.unsqueeze(1) == chunk.unsqueeze(0)).all(-1)
                counts += matches.sum(dim=1)  # 每个候选坐标与该chunk的匹配数
        else:
            # 非分块模式: 一次性计算所有匹配 (显存需求大)
            matches = (selected_grid_coords_unique.unsqueeze(1) == grid_coords.unsqueeze(0)).all(-1)
            counts = matches.sum(dim=1)

        remove_duplicates = counts >= num_overlap  # 匹配数 >= 阈值 → 标记为重复

        return remove_duplicates
    
    # ==========================================================================
    # LOD映射 — 连续LOD值 → 离散整数层级
    # ==========================================================================
    def map_to_int_level(self, pred_level, cur_level):
        """将连续的LOD预测值映射为离散的整数层级。
        
        LOD公式 L* = log2(d_max/d)/log2(fork) + ΔL 产生连续浮点数,
        需要映射到整数层级 [0, cur_level] 并做截断。
        
        支持四种映射策略 (由 self.dist2level 配置):
          - 'floor':       向下取整, 产生最少的高精度层 (保守策略)
          - 'round':       四舍五入, 平衡策略
          - 'ceil':        向上取整, 产生最多的高精度层 (激进策略)
          - 'progressive': 向下取整 + 额外保存小数部分用于平滑过渡
                           _prog_ratio = frac(L*+1) 作为临界层不透明度的衰减系数
                           transition_mask = (锚点层级 == int_level) 标记临界层锚点
        
        参数:
            pred_level: [N] float, 每个锚点的连续LOD预测值 L*
            cur_level:  int, 当前允许的最高层级 (受渐进训练限制)
        
        返回:
            int_level: [N] int, 离散层级, 范围 [0, cur_level]
        
        副作用 (仅progressive模式):
            self._prog_ratio:    [N, 1] 小数部分, 用于不透明度平滑过渡
            self.transition_mask: [N] bool, 标记处于层级临界的锚点
        """
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            # progressive模式: 向下取整, 同时保存小数部分用于平滑过渡
            # +1.0偏移: 使 L*=0 映射到 level=0 (而不是 level=-1)
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            # _prog_ratio 和 transition_mask 仅在 set_anchor_mask (全量锚点) 时需要,
            # weed_out 调用时 pred_level 只覆盖候选子集, 尺寸与 self._level 不匹配, 跳过
            if int_level.shape[0] == self._level.view(-1).shape[0]:
                # _prog_ratio: 小数部分, 在 generate_neural_gaussians() 中
                # 作为临界层锚点不透明度的过渡系数, 实现层级边界的软过渡
                self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)  # [N, 1]
                # transition_mask: 标记 "锚点层级 == 当前LOD等级" 的临界锚点
                self.transition_mask = (self._level.view(-1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    # ==========================================================================
    # MLP Checkpoint (基类空实现, 子类覆盖)
    # ==========================================================================
    def save_mlp_checkpoints(self, path):
        """保存MLP权重。基类空实现, GaussianLoDModel 会覆盖为TorchScript导出。"""
        return

    def load_mlp_checkpoints(self, path):
        """加载MLP权重。基类空实现, GaussianLoDModel 会覆盖为TorchScript加载。"""
        return

    # ==========================================================================
    # 内存清理
    # ==========================================================================
    def clean(self):
        """释放致密化统计量占用的GPU内存。训练结束后调用。"""
        del self.opacity_accum
        del self.anchor_demon
        del self.offset_gradient_accum
        del self.offset_denom
        torch.cuda.empty_cache()