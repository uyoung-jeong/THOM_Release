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

import os
import sys
from PIL import Image
from typing import NamedTuple
#from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
#    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from .colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

try:
    from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
    from utils.sh_utils import SH2RGB
    from scene.gaussian_model import BasicPointCloud
except ModuleNotFoundError as e:
    from gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
    from gaussian_splatting.utils.sh_utils import SH2RGB
    from .gaussian_model import BasicPointCloud
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement

import torch

class RandCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int 
    delta_polar : np.array
    delta_azimuth : np.array
    delta_radius : np.array
    image_path: str
    image_name: str

class RSceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    test_cameras: list
    ply_path: str

class CameraInfo(NamedTuple):
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
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


#only test_camera
def readCircleCamInfo(path,opt, ply_path=None):
    print("Reading Test Transforms")
    test_cam_infos = GenerateOutCameras(opt,render45 = opt.render_45)
    if ply_path is None:
        ply_path = os.path.join(path, "init_points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = opt.init_num_pts       
        if opt.init_shape == 'sphere':
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts)*0.5
            # We create random points inside the bounds of sphere
            xyz = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]
        elif opt.init_shape == 'box':
            xyz = np.random.random((num_pts, 3)) * 1.0 - 0.5
        elif opt.init_shape == 'rectangle_x':
            xyz = np.random.random((num_pts, 3))
            xyz[:, 0] = xyz[:, 0] * 0.6 - 0.3
            xyz[:, 1] = xyz[:, 1] * 1.2 - 0.6
            xyz[:, 2] = xyz[:, 2] * 0.5 - 0.25
        elif opt.init_shape == 'rectangle_z':
            xyz = np.random.random((num_pts, 3))
            xyz[:, 0] = xyz[:, 0] * 0.8 - 0.4
            xyz[:, 1] = xyz[:, 1] * 0.6 - 0.3
            xyz[:, 2] = xyz[:, 2] * 1.2 - 0.6
        elif opt.init_shape == 'pointe':
            num_pts = int(num_pts/5000)
            xyz,rgb = init_from_pointe(opt.init_prompt)
            xyz[:,1] = - xyz[:,1]
            xyz[:,2] = xyz[:,2] + 0.15
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts)*0.05
            # We create random points inside the bounds of sphere
            xyz_ball = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]expend_dims
            rgb_ball = np.random.random((4096, num_pts, 3))*0.0001
            # import pdb; pdb.set_trace()
            rgb = (np.expand_dims(rgb,axis=1)+rgb_ball).reshape(-1,3)
            xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
            xyz = xyz * 1.0
            num_pts = xyz.shape[0]
        elif opt.init_shape == 'shape':
            num_pts = int(num_pts/5000)
            xyz,rgb = shape(opt.init_prompt)
            xyz[:,1] = - xyz[:,1]
            xyz[:,2] = xyz[:,2] + 0.15
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts)*0.05
            # We create random points inside the bounds of sphere
            xyz_ball = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]expend_dims
            rgb_ball = np.random.random((xyz.shape[0], num_pts, 3))*0.0001
            rgb = (np.expand_dims(rgb,axis=1)+rgb_ball).reshape(-1,3)
            xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
            xyz = xyz * 0.8
            num_pts = xyz.shape[0]
        elif opt.init_shape == 'mano':
            raise NotImplementedError
        elif opt.init_shape == 'scene':
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi        
            radius = np.random.rand(num_pts) + opt.radius_range[-1]*3
            # We create random points inside the bounds of sphere
            xyz = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1) # [B, 3]
        else:
            raise NotImplementedError()
        print(f"Generating random point cloud ({num_pts})...")

        shs = np.random.random((num_pts, 3)) / 255.0

        if opt.init_shape == 'pointe' and opt.use_pointe_rgb:
            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((num_pts, 3)))
            storePly(ply_path, xyz, rgb * 255)
        else:
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = RSceneInfo(point_cloud=pcd,
                           test_cameras=test_cam_infos,
                           ply_path=ply_path)
    return scene_info

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.cos(theta),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()


def GenerateOutCameras(opt, size=8, render45 = False):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar+45])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([opt.default_radius])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius,
                        image_path='', image_name=''))  
        
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar*1.5//3])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([opt.default_radius])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx+size, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius,
                        image_path='', image_name=''))         
    return cam_infos


def readColmapSceneInfo(path, images, eval, llffhold=8):
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

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "RandomCam" : readCircleCamInfo
}