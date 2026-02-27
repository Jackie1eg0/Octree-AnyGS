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

        # 更新所有参数组的学习率(指数衰减)
        gaussians.update_learning_rate(iteration)
        
        # ---------- 随机采样一个训练相机 ----------
        # 每个epoch内不重复采样, 队列耗尽后重新生成
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # ---------- 前向渲染 ----------
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # 【Octree-GS 阶段二: 自适应LOD选择 + 渲染】
        # get_render_func() 根据 base_model 返回 "render" 或 "render_2dgs"
        # 渲染内部流程 (gaussian_renderer/render.py):
        #   1. set_anchor_mask()  → 根据相机距离计算每个锚点的连续LOD等级
        #                           L* = log2(d_max/d) + ΔL, 筛选 level ≤ floor(L*)
        #                           【阶段三联动】渐进训练期间, 层级上限受 coarse_index 限制
        #   2. prefilter_voxel()  → 用gsplat投影, 做视锥剔除(radii>0)
        #   3. generate_neural_gaussians() → MLP解码锚点特征→高斯属性
        #                           【阶段五】若 appearance_dim>0, 拼接外观码后输入Color MLP
        #                           【阶段二】若 dist2level=="progressive", 对临界层opacity做平滑插值
        #   4. gsplat.rasterization() → 高斯光栅化出图像
        render_pkg = getattr(modules, get_render_func(dataset.base_model))(viewpoint_cam, gaussians, pipe, scene.background, iteration, dataset.render_mode)
        image, scaling = render_pkg["render"], render_pkg["scaling"]

        # ======================================================================
        # 损失函数计算
        # ======================================================================
        gt_image = viewpoint_cam.original_image.cuda()  # 真值图像
        
        # 主损失 = (1-λ)*L1 + λ*SSIM, 默认 lambda_dssim=0.2
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
       
        # 缩放正则化: 惩罚高斯体积过大, 默认 lambda_dreg=0.01
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
            #   (默认: start_stat=500, update_from=1500, 
            #    update_until=25000, update_interval=100)
            # ==================================================================
            if iteration < opt.update_until and iteration > opt.start_stat:
                
                # (1) 每步累积梯度统计 (basic_model.py: training_statis)
                #     收集 viewspace_points 的梯度 → offset_gradient_accum
                #     累加 opacity → opacity_accum
                #     计数锚点被渲染次数 → anchor_demon
                #     这些统计量是致密化决策的依据
                gaussians.training_statis(render_pkg, image.shape[2], image.shape[1])
                
                # (2) 每 update_interval(默认100)步执行一次致密化
                #     run_densify() 内部流程 (lod_model.py):
                #     
                #     a) anchor_growing() — 生长新锚点:
                #        遍历每一层 cur_level:
                #        - 计算动态阈值 τ_L = τ_g × (fork^β)^L
                #          (τ_g=0.0002, β=update_ratio=0.2)
                #        - 同层生长: τ_L ≤ grad < τ_{L+1} → 在当前层添加锚点
                #        - 【跨层生长】: grad ≥ τ_{L+1} → 在L+1层添加锚点
                #          ⚠ 仅在渐进训练结束后(iteration > coarse_intervals[-1])才允许!
                #        - 【ΔL更新】: anchor级梯度 > τ_L×extra_ratio(0.25)
                #          → _extra_level += extra_up(0.02)
                #          同样仅在渐进训练结束后才执行
                #        - 每次新增锚点都要经过 weed_out() 可见性检查
                #     
                #     b) prune_anchor() — 剪枝低贡献锚点:
                #        - 条件: opacity_accum < min_opacity(0.005) × anchor_demon
                #          且 anchor_demon > update_interval × success_threshold(0.8)
                #        - 即: 被足够多次渲染但平均不透明度仍然很低的锚点会被删除
                if opt.densification and iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.run_densify(iteration, opt)
            
            # 致密化阶段结束后, 清理临时统计量
            elif iteration == opt.update_until:
                gaussians.clean()
                    
            # ---------- 优化器步进 ----------
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
            # ---------- 保存Checkpoint ----------
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
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

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
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', len(scene.gaussians.get_anchor), iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(base_model, model_path, name, iteration, views, gaussians, pipe, background, render_mode):
    """渲染一组视角并保存图像。
    输出: 渲染图/误差图/GT图 + 每视角可见高斯计数。
    【注意】渲染时会调用与训练相同的LOD选择流程(set_anchor_mask等)。
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    per_view_dict = {}
    modules = __import__('gaussian_renderer')
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        render_pkg = getattr(modules, get_render_func(base_model))(view, gaussians, pipe, background, iteration, render_mode)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
        
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset, opt, pipe, iteration, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    """渲染训练集/测试集并计算FPS。
    注意: 会重新加载模型并调用 set_coarse_interval() 以确保渲染时LOD选择正确。
    """
    with torch.no_grad():
        modules = __import__('scene.gs_model_'+dataset.base_model, fromlist=[''])
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
        gaussians.eval()
        gaussians.set_coarse_interval(opt)
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count = render_set(dataset.base_model, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, dataset.render_mode)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/train_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.base_model, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, dataset.render_mode)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    """对渲染结果计算质量指标: PSNR, SSIM, LPIPS, 以及平均可见高斯数。
    结果保存到 results.json 和 per_view.json。
    """

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        logger.info("  GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(visible_count).float().mean(), ".5"))
        print("")

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
        
        full_dict[scene_dir][method].update({
            "PSNR": torch.tensor(psnrs).mean().item(),
            "SSIM": torch.tensor(ssims).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
            "GS_NUMS": torch.tensor(visible_count).float().mean().item(),
            })

        per_view_dict[scene_dir][method].update({
            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}
            })

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

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
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')  # 核心: YAML配置文件路径(如base_model.yaml或lod_model.yaml),决定了使用哪个模型
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)  # Warmup模式: 训练两轮, 第二轮用第一轮输出的点云进行重新初始化
    parser.add_argument('--use_wandb', action='store_true', default=False)  # 是否使用wandb进行数据记录、绘制图表
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])  # 在哪些迭代步数进行测试集评估,比如[10K,20K,30K,40K]显示PSNR SSIM指标
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])  # 在哪些迭代步数保存.ply点云文件
    parser.add_argument("--quiet", action="store_true")  
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 保存CheckPoint(完整模型状态的迭代步数)
    parser.add_argument("--start_checkpoint", type=str, default = None)  # 从哪个CheckPoint文件恢复训练
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

    # 构建Output文件路径并备份config.yaml
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lp.model_path = os.path.join("outputs", lp.dataset_name, lp.scene_name, cur_time)
    os.makedirs(lp.model_path, exist_ok=True)
    shutil.copy(args.config, os.path.join(lp.model_path, "config.yaml"))

    logger = get_logger(lp.model_path)

    # 设置默认的测试/保存时机(每10000步一次)
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
    # 执行训练(Octree-AnyGS的核心训练函数)
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
    # 渲染 + 评估
    # ======================================================================
    logger.info(f'\nStarting Rendering~')
    if lp.eval:
        visible_count = render_sets(lp, op, pp, -1, skip_train=True, skip_test=False, wandb=wandb, logger=logger)
    else:
        visible_count = render_sets(lp, op, pp, -1, skip_train=False, skip_test=True, wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    logger.info("\n Starting evaluation...")
    eval_name = 'test' if lp.eval else 'train'
    evaluate(lp.model_path, eval_name, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
