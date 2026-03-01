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

┌──────────────────────────────────────────────────────────────────┐
│                        BasicModel (基类)                        │
├──────────────────────────────────────────────────────────────────┤
│ 1. 激活函数设置: setup_functions()                                │
│ 2. 优化器管理:                                                    │
│    - replace_tensor_to_optimizer(): 替换优化器中的参数              │
│    - cat_tensors_to_optimizer():    向优化器追加新参数(锚点生长)    │
│    - _prune_anchor_optimizer():     从优化器移除参数(锚点剪枝)     │
│ 3. 训练统计: training_statis()  收集梯度/不透明度统计量             │
│ 4. 去重工具: get_remove_duplicates()  体素网格坐标去重              │
│ 5. LOD映射: map_to_int_level()  连续LOD→离散层级映射               │
│ 6. 默认空实现: eval/train/set_appearance 等(子类可覆盖)           │
└──────────────────────────────────────────────────────────────────┘

子类关系:
  BasicModel (本文件)
    ├── GaussianModel      (gs_model_scaffoldgs/model.py)  — 原版Scaffold-GS
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
        self.scaling_activation = torch.exp              # 缩放: exp确保正数
        self.scaling_inverse_activation = torch.log      # 缩放反函数: log
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

    # ==========================================================================
    # 训练统计收集 — 为致密化(阶段四)提供决策依据
    # 每个训练步的渲染结束后调用, 累积梯度和不透明度统计
    # ==========================================================================
    def training_statis(self, render_pkg, width, height):
        """收集每帧渲染的梯度和不透明度统计信息, 用于指导致密化决策。
        
        在 train.py 的主循环中, 每次渲染后调用。收集三类统计量:
          1. opacity_accum:         每个锚点的不透明度累积 (用于剪枝判断)
          2. anchor_demon:          每个锚点被渲染的次数 (用于归一化)
          3. offset_gradient_accum: 每个神经高斯的屏幕空间梯度累积 (用于生长判断)
        
        实现逻辑:
          Step 1: 从 render_pkg 中提取渲染结果
          Step 2: 累积每个锚点下所有offset的不透明度总和 → opacity_accum
          Step 3: 可见锚点的渲染计数 +1 → anchor_demon
          Step 4: 计算屏幕空间梯度范数, 累积到 offset_gradient_accum
        
        参数:
            render_pkg: dict, 渲染器返回的结果包, 包含:
                viewspace_points:  [1, M, 2] 屏幕空间坐标 (含梯度)
                visibility_filter: [M] bool, 渲染时可见的高斯掩码
                visible_mask:      [N] bool, 可见锚点掩码 (来自prefilter)
                selection_mask:    [N*k] bool, 有效神经高斯掩码 (opacity>0)
                opacity:           [M, 1] 渲染的不透明度值
            width:  渲染图像宽度 (用于梯度缩放)
            height: 渲染图像高度 (用于梯度缩放)
        """
        viewspace_point_tensor = render_pkg["viewspace_points"]
        update_filter = render_pkg["visibility_filter"]     # [M] 实际被渲染的高斯
        anchor_visible_mask = render_pkg["visible_mask"]     # [N] 可见锚点
        offset_selection_mask = render_pkg["selection_mask"] # [N*k] 有效高斯 (opacity>0)
        opacity = render_pkg["opacity"]                      # [M, 1] 不透明度
        
        # ===== Step 2: 累积不透明度 =====
        # 将有效高斯的opacity填回全尺寸数组, 然后按锚点求和
        temp_opacity = torch.zeros(offset_selection_mask.shape[0], dtype=torch.float32, device="cuda")
        temp_opacity[offset_selection_mask] = opacity.clone().view(-1).detach()
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])  # [N, k] 按锚点分组
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)  # 按锚点求和
        
        # ===== Step 3: 累积锚点渲染次数 =====
        self.anchor_demon[anchor_visible_mask] += 1

        # ===== Step 4: 累积屏幕空间梯度 =====
        # 构建combined_mask: 标记哪些offset位置对应实际被渲染且可见的高斯
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)  # [N*k]
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask  # 先标记"有效高斯"的位置
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter  # 再从中筛选"实际被渲染"的
        
        # 计算屏幕空间梯度范数 (缩放到像素坐标)
        grad = viewspace_point_tensor.grad.squeeze(0)  # [M, 2] 屏幕空间坐标梯度
        grad[:, 0] *= width * 0.5    # NDC → 像素坐标缩放
        grad[:, 1] *= height * 0.5
        grad_norm = torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)  # [M', 1]
        self.offset_gradient_accum[combined_mask] += grad_norm  # 累积梯度范数
        self.offset_denom[combined_mask] += 1                    # 累积计数
        
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

        remove_duplicates = counts >= num_overlap  # 匹配数 >= 阈值(默认为1) → 标记为重复

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
            # _prog_ratio: 小数部分, 在 generate_neural_gaussians() 中
            # 作为临界层锚点不透明度的过渡系数, 实现层级边界的软过渡
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)  # [N, 1]
            # transition_mask: 标记 "锚点层级 == 当前LOD等级" 的临界锚点
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
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