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
scene/dataset_readers.py — 场景数据读取器

本文件负责从不同格式的数据(如colmap、blender、city)中读取相机参数、图像和初始点云, 
并将数据统一封装为 SceneInfo 结构供 Scene 类使用。

支持三种数据格式:
┌─────────────────────────────────────────────────────────────────────┐
│  Colmap 格式 (真实场景, 如 Mip-NeRF360, Tanks&Temples)              │
│    数据目录:  source_path/sparse/0/ (images.bin + cameras.bin)      │
│    点云来源:  COLMAP SfM 重建的稀疏点云 (points3D.bin)              │
│    调用入口:  readColmapSceneInfo()                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Blender 格式 (合成场景, 如 NeRF Symbolic)                          │
│    数据目录:  source_path/transforms_train.json + transforms_test   │
│    点云来源:  随机生成 或 已有PLY文件                                │
│    调用入口:  readNerfSyntheticInfo()                               │
├─────────────────────────────────────────────────────────────────────┤
│  City 格式 (城市大场景)                                              │
│    数据目录:  source_path/transforms.json + *.ply 或 LAS/*.las     │
│    点云来源:  PLY文件 或 LAS点云文件                                 │
│    调用入口:  readCityInfo()                                        │
└─────────────────────────────────────────────────────────────────────┘

数据流:
  数据集文件 → readXxxSceneInfo() → SceneInfo(点云, 相机列表, 场景范围)
                                         ↓
                                  Scene.__init__() 中使用
"""

import os
import glob
import sys
import cv2
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from utils.graphics_utils import BasicPointCloud
import concurrent.futures


# =============================================================================
# 数据结构定义
# =============================================================================

class CameraInfo(NamedTuple):
    """单个相机的完整信息 (中间数据结构, 尚未加载图像到GPU)。
    
    由 readColmapCameras() 或 readCamerasFromTransforms() 生成,
    后续在 cameraList_from_camInfos() 中被转换为 Camera 对象 (包含GPU张量)。
    
    属性:
        uid:        int,      相机唯一ID
        R:          [3, 3],   旋转矩阵 (W2C的旋转部分的转置, 注意是转置存储!)
                              实际 W2C 旋转 = R.T, 这是CUDA代码中glm库的约定
        T:          [3],      平移向量 (W2C的平移部分)  ===> 具有R和T可以构建W2C矩阵
        FovY:       float,    垂直视场角 (弧度)
        FovX:       float,    水平视场角 (弧度)
        image:      PIL.Image, 原始图像 (CPU, 尚未转为Tensor)
        image_path: str,      图像文件路径
        image_name: str,      图像名称 (不含扩展名)
        width:      int,      图像宽度 (像素)
        height:     int,      图像高度 (像素)
    """
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    """场景完整信息 (由各 readXxxSceneInfo 函数返回, 传递给 Scene.__init__)。
    
    属性:
        point_cloud:       BasicPointCloud, 初始SfM点云 (points[M,3], colors[M,3], normals[M,3])
        train_cameras:     [CameraInfo], 训练相机列表
        test_cameras:      [CameraInfo], 测试相机列表
        nerf_normalization: dict, 场景归一化参数(getNerfppNorm函数计算出nerf_normalization字典):
                            - "translate": [3], 场景中心的平移向量 (将相机中心移到原点)
                            - "radius":    float, 场景半径 (所有相机到中心的最大距离 × 1.1)
        ply_path:          str, 点云PLY文件路径
    """
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


# =============================================================================
# 场景归一化工具
# =============================================================================

def getNerfppNorm(cam_info):
    """计算 NeRF++ 风格的场景归一化参数。
    根据所有训练相机的世界坐标, 计算场景中心和半径。
    
    ===================== 3DGS借用NerF++的归一化方式=====================
    但是没有任何地方把translate应用到点云坐标或相机位置来做居中,也没有用radius除坐标做缩放
    仅仅使用其中的radius作为场景尺度度量,用于自适应调整学习率,没有真正对场景坐标归一化变换
    ========================================================================

    用于:
      1. Scene.cameras_extent → spatial_lr_scale (位置学习率缩放)
      2. 大场景下的坐标归一化
    
    计算流程:
      1. 从每个相机的 R, T 恢复 W2C → C2W, 提取相机世界坐标
      2. 计算所有相机中心的平均值 → 场景中心
      3. 计算每个相机到中心的距离, 取最大值 → 场景对角线
      4. 半径 = 对角线 × 1.1 (留10%余量)
    
    参数:
        cam_info: [CameraInfo], 训练相机信息列表
    
    返回:
        dict:
            "translate": [3] ndarray, -center (将场景中心移到原点的平移向量)
            "radius":    float, 场景半径 (cameras_extent)
    """
    def get_center_and_diag(cam_centers):
        """内部辅助: 计算所有Camera中心的平均值和最大离散距离(平均中心到最远Camera的距离)
        
        参数:
            cam_centers: list of [3, 1] ndarray, 每个相机的世界坐标
        
        返回:
            center:   [3] ndarray, 相机中心的均值
            diagonal: float, 某个相机到中心的最大距离
        """
        cam_centers = np.hstack(cam_centers)                    # [3, N] 拼接所有相机中心
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)  # [3, 1] 得到所有相机中心的平均值
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)  # [1, N] 每个相机到平均中心的距离
        diagonal = np.max(dist)                                 # 最大距离(最远Camera到平均中心的距离)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)      # [4, 4] W2C矩阵 (从R的转置和T恢复)
        C2W = np.linalg.inv(W2C)                # [4, 4] C2W矩阵 = W2C的逆
        cam_centers.append(C2W[:3, 3:4])        # [3, 1] Camera世界坐标 = C2W的平移列

    center, diagonal = get_center_and_diag(cam_centers) # 得到所有Camera在世界坐标下的平均中心和最大离散距离(平均中心到最远Camera的距离)
    radius = diagonal * 1.1                     # 留 10% 余量

    translate = -center                         # 平移向量: 将所有Camera的平均中心移到原点

    return {"translate": translate, "radius": radius}


# =============================================================================
# PLY 点云文件读写
# =============================================================================

def fetchPly(path):
    """从PLY文件读取点云数据。
    
    读取顶点的 (x, y, z) 坐标、(r, g, b) 颜色和 (nx, ny, nz) 法线。
    如果PLY文件中缺少颜色或法线属性, 则用随机值填充。
    
    参数:
        path: str, PLY文件路径
    
    返回:
        BasicPointCloud:
            points:  [M, 3] ndarray, 点云坐标
            colors:  [M, 3] ndarray, 颜色值 (归一化到 [0, 1])
            normals: [M, 3] ndarray, 法线向量
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0    # uint8 → [0,1]
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])     # 无颜色属性时随机填充
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])    # 无法线属性时随机填充
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    """将点云数据保存为PLY文件。
    
    参数:
        path: str, 输出PLY文件路径
        xyz:  [M, 3] ndarray, 点云坐标
        rgb:  [M, 3] ndarray, 颜色值 (0-255 uint8 范围)
    """
    # PLY结构化数组: 每个点 = (x, y, z, nx, ny, nz, r, g, b)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)    # 法线全零 (仅保存坐标和颜色)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# =============================================================================
# COLMAP 格式相机读取
# =============================================================================

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    """从 COLMAP 的外参和内参数据中构建 CameraInfo 列表。
    
    使用多线程并行加载图像以加速 I/O。
    
    支持三种COLMAP相机模型:
      - SIMPLE_PINHOLE:  单焦距 (fx = fy)
      - SIMPLE_RADIAL:   单焦距 + 径向畸变 (fx = fy, 这里忽略畸变)
      - PINHOLE:         双焦距 (fx ≠ fy)
    
    参数:
        cam_extrinsics: dict, COLMAP外参 (key: image_id, value: Image对象)
                        包含旋转四元数 qvec 和平移向量 tvec
        cam_intrinsics: dict, COLMAP内参 (key: camera_id, value: Camera对象)
                        包含焦距、主点、图像尺寸
        images_folder:  str, 图像文件夹路径
    
    返回:
        [CameraInfo], 按 image_name 排序的相机信息列表
    """
    cam_infos = []
    
    def process_frame(idx, key):
        """处理单个相机帧: 提取相机内外参 + 加载图像。"""
        extr = cam_extrinsics[key]              # 相机位姿
        intr = cam_intrinsics[extr.camera_id]   # 相机内参
        height = intr.height
        width = intr.width

        uid = intr.id
        # COLMAP 的旋转四元数 → 旋转矩阵, 再转置 (CUDA glm库约定)
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # 根据相机模型计算视场角 (FoV)---在3DGS中具有核心作用:构建透视投影矩阵(Project Matrix),决定相机能看到多大角度范围+视锥裁剪
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            # 单焦距模型: fx = fy = params[0]
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)   # FoV = 2 * arctan(size / (2*focal))
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            # 双焦距模型: fx = params[0], fy = params[1]
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        # CameraInfo 包含了相机位姿、内参、视场角、图像路径、图像名称、图像尺寸(所存即所得)
        return CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)

    # 使用线程池并行加载图像 (I/O密集型, 线程池比进程池更合适)
    # 多线程并行加载所有相机图像
    ct = 0
    progress_bar = tqdm(cam_extrinsics, desc="Loading dataset")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, idx, key) for idx, key in enumerate(cam_extrinsics)]

        for future in concurrent.futures.as_completed(futures):
            cam_info = future.result()
            cam_infos.append(cam_info)
            
            ct+=1
            if ct % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(cam_extrinsics)}"+Style.RESET_ALL})
                progress_bar.update(10)

        progress_bar.close()

    # 按图像名排序,Mip-NERF的images一般都带编号, 确保顺序一致性 (多线程完成顺序不确定)
    cam_infos = sorted(cam_infos, key = lambda x : x.image_name)
    return cam_infos


# =============================================================================
# Blender/City 格式相机读取 (基于 transforms.json)
# =============================================================================

def readCamerasFromTransforms(path, transformsfile, extension=".png"):
    """从 NeRF 风格的 transforms JSON 文件中读取相机参数。
    
    JSON 文件格式 (Blender 导出):
      {
        "camera_angle_x": float,          // 水平FoV (弧度), 可选
        "frames": [
          {
            "file_path": "train/r_0",     // 相对图像路径
            "transform_matrix": [[...]]   // [4,4] Camera-to-World 矩阵 (OpenGL约定)
            "fl_x": float, "fl_y": float  // 可选: 像素焦距 (当 camera_angle_x 不存在时使用)
          }, ...
        ]
      }
    
    坐标系转换:
      OpenGL/Blender: Y轴朝上, Z轴朝后 (右手系)
      COLMAP:         Y轴朝下, Z轴朝前
      转换方法: C2W[:3, 1:3] *= -1 (翻转Y和Z轴)
    
    参数:
        path:           str, 数据集根目录
        transformsfile: str, JSON文件名 (如 "transforms_train.json")
        extension:      str, 图像扩展名 (默认 ".png", 如果文件名已包含扩展名则自动清空)
    
    返回:
        [CameraInfo], 按 image_name 排序的相机信息列表
    """
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]       # 全局水平FoV (所有帧共享)
        except:
            fovx = None                              # 某些数据集没有全局FoV, 每帧各自指定焦距

        frames = contents["frames"]
        # 检查文件名是否已包含扩展名 (避免重复添加)
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        def process_frame(idx, frame):
            """处理单帧: 坐标系转换 + 分解R/T + 加载图像。"""
            cam_name = frame["file_path"] + extension
            image_path = os.path.join(path, cam_name)
            if not os.path.exists(image_path):
                raise ValueError(f"Image {image_path} does not exist!")
            
            # NeRF 的 transform_matrix 是 Camera-to-World (C2W) 矩阵
            c2w = np.array(frame["transform_matrix"])

            # 坐标系转换: OpenGL/Blender → COLMAP
            # OpenGL: Y朝上, Z朝后; COLMAP: Y朝下, Z朝前
            # 翻转 Y 和 Z 轴 (第1列和第2列取反)
            c2w[:3, 1:3] *= -1

            # C2W → W2C, 再分解为 R 和 T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])   # 转置存储 (CUDA glm库约定)
            T = w2c[:3, 3]

            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            # 计算 FoV
            if fovx is not None:
                # 有全局 camera_angle_x: 用它计算 FovX, 再换算 FovY
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx
            else:
                # 无全局 FoV: 每帧各自提供像素焦距 fl_x, fl_y
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            return CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        
        # 多线程并行加载
        ct = 0
        progress_bar = tqdm(frames, desc="Loading dataset")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_frame, idx, frame) for idx, frame in enumerate(frames)]

            for future in concurrent.futures.as_completed(futures):
                cam_info = future.result()
                cam_infos.append(cam_info)
                
                ct+=1
                if ct % 10 == 0:
                    progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                    progress_bar.update(10)

            progress_bar.close()
    
    cam_infos = sorted(cam_infos, key = lambda x : x.image_name)
    return cam_infos


# =============================================================================
# 场景读取入口 — COLMAP 格式
# =============================================================================

def readColmapSceneInfo(path, images, eval, llffhold=8, warmup_ply_path=None):
    """从 COLMAP 格式的数据集读取完整场景信息。
    
    COLMAP 数据目录结构:
      source_path/
        ├── images/               ← 输入图像
        ├── sparse/0/
        │   ├── cameras.bin (.txt)  ← 相机内参
        │   ├── images.bin  (.txt)  ← 相机外参 (位姿)
        │   └── points3D.bin(.txt)  ← SfM稀疏点云
        └── ...
    
    参数:
        path:            str, 数据集根目录 (source_path)
        images:          str, 图像子目录名 (默认 "images")
        eval:            bool, 是否分离训练/测试集
                         True:  每 llffhold 张取1张作为测试集 (LLFF协议)
                         False: 所有图像用于训练, 测试集为空
        llffhold:        int, LLFF测试集间隔 (默认8, 即每8张取1张做测试)
        warmup_ply_path: str 或 None, 外部PLY点云路径 (覆盖COLMAP的点云)
    
    返回:
        SceneInfo: 包含点云、训练/测试相机、场景范围的完整场景数据
    """
    # 优先尝试二进制格式 (更快), 失败则回退到文本格式
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 读取Colmap输出所有相机信息 (外参+内参 → CameraInfo 列表)
    reading_dir = images
    cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir))
    
    # 训练/测试集划分 (LLFF协议: 每 llffhold 张取1张做测试)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # 计算场景归一化参数 (相机中心 → 场景中心和半径)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 读取初始SfM点云
    if warmup_ply_path is not None:
        # 使用外部提供的PLY文件 (如预训练模型的点云)
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        # 使用 COLMAP 的 SfM 点云
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            # 首次打开: 将 points3D.bin → points3D.ply (只需转换一次)
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    # 封装为 SceneInfo 返回
    # 点云pcd 训练的相机视图 train_cameras 测试的相机视图 test_cameras 归一化参数 nerf_normalization ply文件路径
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# =============================================================================
# 场景读取入口 — Blender/NeRF Synthetic 格式(用Colmap不用看)
# =============================================================================

def readNerfSyntheticInfo(path, eval, extension=".png", warmup_ply_path=None):
    """从 Blender/NeRF Synthetic 格式的数据集读取完整场景信息。
    
    Blender 数据目录结构:
      source_path/
        ├── transforms_train.json   ← 训练集相机参数
        ├── transforms_test.json    ← 测试集相机参数
        ├── train/                  ← 训练图像
        ├── test/                   ← 测试图像
        └── points3d.ply (可选)     ← 如果有则使用, 否则随机生成
    
    与 COLMAP 格式的区别:
      1. 训练/测试集已在 JSON 文件中分好
      2. 可能没有 SfM 点云, 需要随机生成初始点
    
    参数:
        path:            str, 数据集根目录
        eval:            bool, 是否分离训练/测试集
                         True:  使用 transforms_test.json 中的相机作为测试集
                         False: 训练集 = train + test 合并, 测试集为空
        extension:       str, 图像扩展名 (默认 ".png")
        warmup_ply_path: str 或 None, 外部PLY文件路径
    
    返回:
        SceneInfo: 完整场景数据
    """
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", extension)
    
    if not eval:
        # 非评估模式: 合并训练+测试集用于训练
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    if warmup_ply_path is not None:
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_paths = glob.glob(os.path.join(path, "*.ply"))
        if len(ply_paths)==0:
            # 没有现成的PLY文件: 随机生成初始点云
            # Blender 合成场景通常没有 SfM 重建, 需要随机初始化
            ply_path = os.path.join(path, "points3d.ply")
            num_pts = 10_000
            print(f"Generating random point cloud ({num_pts})...")
            # 在 Blender 场景的典型范围 [-1.3, 1.3]³ 内随机采样
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            colors = np.random.random((num_pts, 3))
            normals=np.zeros((num_pts, 3))
            pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

            storePly(ply_path, xyz, colors*255)     # 保存为PLY (颜色转为0-255)
        else:
            # 有现成的PLY文件: 直接加载
            ply_path = ply_paths[0]
            pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# =============================================================================
# 场景读取入口 — City 大场景格式(用Colmap不用看)
# =============================================================================

def readCityInfo(path, eval, llffhold=8, extension=".png", warmup_ply_path=None):
    """从 City 格式的数据集读取完整场景信息。
    
    City 数据目录结构:
      source_path/
        ├── transforms.json         ← 相机参数 (单个文件, 不区分训练/测试)
        ├── *.ply                   ← 点云文件
        └── LAS/*.las (备选)        ← LAS格式点云 (如果没有PLY)
    
    与其他格式的区别:
      1. 只有一个 transforms.json, 训练/测试划分沿用 LLFF 协议
      2. 点云来源优先PLY, 备选LAS (城市测绘常用的点云格式)
    
    参数:
        path:            str, 数据集根目录
        eval:            bool, 是否分离训练/测试集
        llffhold:        int, LLFF测试集间隔 (默认8)
        extension:       str, 图像扩展名
        warmup_ply_path: str 或 None, 外部PLY文件路径 (本函数未使用, 保留接口一致性)
    
    返回:
        SceneInfo: 完整场景数据
    """
    json_path = glob.glob(os.path.join(path, f"transforms.json"))[0].split('/')[-1]
    print("Reading Training Transforms from {}".format(json_path))
    
    # 加载点云: 优先PLY, 备选LAS
    ply_path = glob.glob(os.path.join(path, "*.ply"))[0]
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except:
            raise ValueError("must have tiepoints!")
    else:
        # 读取 LAS 格式点云 (城市测绘数据常用)
        las_paths = glob.glob(os.path.join(path, "LAS/*.las"))
        las_path = las_paths[0]
        print(f'las_path: {las_path}')
        try:
            pcd = read_multiple_las_files(las_paths, ply_path)
        except:
            raise ValueError("Load LAS failed!")
    
    # 读取相机
    cam_infos = readCamerasFromTransforms(path, json_path, extension)
    
    print("Load Cameras: ", len(cam_infos))
    train_cam_infos = []
    test_cam_infos = []
    
    # 训练/测试划分: 沿用 LLFF 协议 (与 COLMAP 格式一致)
    if not eval:
        train_cam_infos.extend(cam_infos)
        test_cam_infos = []
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# =============================================================================
# 场景读取回调注册表 — Scene.__init__() 根据数据格式查表调用
# =============================================================================
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,       # source_path/sparse/ 存在时调用
    "Blender": readNerfSyntheticInfo,    # source_path/transforms_train.json 存在时调用
    "City": readCityInfo                 # source_path/transforms.json 存在时调用
}