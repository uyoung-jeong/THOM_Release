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
import torch
import random
import torch.nn.functional as F
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import math
import json
from pathlib import Path
from utils.pointe_utils import init_from_pointe
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH
from utils.general_utils import inverse_sigmoid_np
from scene.gaussian_model import BasicPointCloud
import open3d as o3d
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
"""
import tyro
import rembg
from kiui.op import recenter
import torchvision.transforms.functional as TF
from lgm.models import LGM
from lgm.options import AllConfigs
from mvdream.pipeline_mvdream import MVDreamPipeline
from safetensors.torch import load_file

def run_lgm(prompt, output_ply_path):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    checkpoint_path = './load/model_fp16_fixrot.safetensors'

    lgm_opt = tyro.cli(AllConfigs, args=['big'])
    lgm_opt.resume = checkpoint_path
    #lgm_opt = AllConfigs
    model = LGM(lgm_opt)

    ckpt = load_file(checkpoint_path, device='cpu')
    model.load_state_dict(ckpt, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.half().to(device)
    model.eval()

    tan_half_fov = np.tan(0.5 * np.deg2rad(lgm_opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (lgm_opt.zfar + lgm_opt.znear) / (lgm_opt.zfar - lgm_opt.znear)
    proj_matrix[3, 2] = - (lgm_opt.zfar * lgm_opt.znear) / (lgm_opt.zfar - lgm_opt.znear)
    proj_matrix[2, 3] = 1

    pipe_text = MVDreamPipeline.from_pretrained(
        'ashawkey/mvdream-sd2.1-diffusers', # remote weights
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # local_files_only=True,
    )
    pipe_text = pipe_text.to(device)

    bg_remover = rembg.new_session()

    # process text prompt
    prompt_neg = 'ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate'
    input_elevation = 0
    input_num_steps = 30
    mv_image_uint8 = pipe_text(prompt, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=7.5, elevation=input_elevation)
    mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)
    # bg removal
    mv_image = []
    for i in range(4):
        image = rembg.remove(mv_image_uint8[i], session=bg_remover) # [H, W, 4]
        # to white bg
        image = image.astype(np.float32) / 255
        image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
        image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
        mv_image.append(image)
    
    mv_image_grid = np.concatenate([
        np.concatenate([mv_image[1], mv_image[2]], axis=1),
        np.concatenate([mv_image[3], mv_image[0]], axis=1),
    ], axis=0)

    # generate gaussians
    input_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(lgm_opt.input_size, lgm_opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    rays_embeddings = model.prepare_default_rays(device, elevation=input_elevation)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
            #gaussians.shape: [1, 65536, 14]
    model.gs.save_ply(gaussians, output_ply_path)

    xyz = gaussians[0,:,:3].cpu().numpy()
    rgb = gaussians[0,:,11:].cpu().numpy()
    return xyz, rgb
"""
def add_points(coords,rgb):
    pcd_by3d = o3d.geometry.PointCloud()
    pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
    

    bbox = pcd_by3d.get_axis_aligned_bounding_box()
    np.random.seed(0)

    num_points = 1000000  
    points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))


    kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)


    points_inside = []
    color_inside= []
    for point in points:
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
        if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
            points_inside.append(point)
            color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

            
            

    all_coords = np.array(points_inside)
    all_rgb = np.array(color_inside)
    all_coords = np.concatenate([all_coords,coords],axis=0)
    all_rgb = np.concatenate([all_rgb,rgb],axis=0)
    return all_coords,all_rgb

def shape(prompt):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
    diffusion = diffusion_from_config_shape(load_config('diffusion'))

    batch_size = 1
    guidance_scale = 15.0
    prompt = str(prompt)
    print('prompt',prompt)

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    render_mode = 'nerf' # you can change this to 'stf'
    size = 512 # this is the size of the renders; higher values take longer to render.

    cameras = create_pan_cameras(size, device)

    # self.shapeimages = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)

    pc = decode_latent_mesh(xm, latents[0]).tri_mesh()

    skip = 1
    coords = pc.verts
    rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

    coords = coords[::skip]
    rgb = rgb[::skip]
    return coords,rgb


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


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


class RSceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    test_cameras: list
    ply_path: str

# def getNerfppNorm(cam_info):
#     def get_center_and_diag(cam_centers):
#         cam_centers = np.hstack(cam_centers)
#         avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
#         center = avg_cam_center
#         dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
#         diagonal = np.max(dist)
#         return center.flatten(), diagonal

#     cam_centers = []

#     for cam in cam_info:
#         W2C = getWorld2View2(cam.R, cam.T)
#         C2W = np.linalg.inv(W2C)
#         cam_centers.append(C2W[:3, 3:4])

#     center, diagonal = get_center_and_diag(cam_centers)
#     radius = diagonal * 1.1

#     translate = -center

#     return {"translate": translate, "radius": radius}



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
def readCircleCamInfo(path,opt):
    #print("Reading Test Transforms")
    test_cam_infos = GenerateOutCameras(opt,render45 = opt.render_45)
    ply_path = os.path.join(path, "init_points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = opt.init_num_pts       
        shs = np.random.random((num_pts, 3)) / 255.0
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
            """
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
            """
            xyz,rgb = shape(opt.init_prompt)
            xyz[:,1] = - xyz[:,1]
            xyz[:,2] = xyz[:,2] + 0.15

            num_init_pts = xyz.shape[0]
            
            #print(f"num_init_pts: {num_init_pts}, num_pts: {num_pts}")

            if num_init_pts < num_pts: # needs oversampling
                n_multi = math.ceil(num_pts/num_init_pts)
                #print(f"n_multi: {n_multi}")
                thetas = np.random.rand(n_multi)*np.pi
                phis = np.random.rand(n_multi)*2*np.pi        
                radius = np.random.rand(n_multi)*0.05
                # We create random points inside the bounds of sphere
                xyz_ball = np.stack([
                    radius * np.sin(thetas) * np.sin(phis),
                    radius * np.sin(thetas) * np.cos(phis),
                    radius * np.cos(thetas),
                ], axis=-1) # [B, 3]expend_dims

                rgb_ball = np.random.random((xyz.shape[0], n_multi, 3))*0.0001
                rgb = (np.expand_dims(rgb,axis=1)+rgb_ball).reshape(-1,3)
                xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
                num_init_pts = xyz.shape[0]
                #print(f"oversampled num_init_pts: {num_init_pts}")
            
            if num_init_pts > num_pts*2: # randomly choose points
                indices = np.arange(num_init_pts)
                chosen_ids = np.random.choice(indices, num_pts*2)
                xyz = xyz[chosen_ids]
                rgb = rgb[chosen_ids]
            
            xyz = xyz * 0.8
            num_pts = xyz.shape[0]
            shs = RGB2SH(rgb)
            #print(f"final num_pts: {num_pts}")
        
        #elif opt.init_shape == 'lgm':
            #lgm_ply_path = ply_path.replace('init_points3d.ply', 'init_lgm.ply')
            #xyz, rgb = run_lgm(opt.init_prompt, lgm_ply_path)
            #num_pts = xyz.shape[0]
        
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
        elif opt.init_shape == 'mano':
            template_mano_path = 'load/mano_right.ply'
            plydata = PlyData.read(template_mano_path)
            
            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
            num_pts = xyz.shape[0]
            shs = np.random.random((num_pts, 3)) / 255.0
        else:
            raise NotImplementedError()

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
#borrow from https://github.com/ashawkey/stable-dreamfusion

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

# def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

#     theta = theta / 180 * np.pi
#     phi = phi / 180 * np.pi
#     angle_overhead = angle_overhead / 180 * np.pi
#     angle_front = angle_front / 180 * np.pi

#     centers = torch.stack([
#         radius * torch.sin(theta) * torch.sin(phi),
#         radius * torch.cos(theta),
#         radius * torch.sin(theta) * torch.cos(phi),
#     ], dim=-1) # [B, 3]

#     # lookat
#     forward_vector = safe_normalize(centers)
#     up_vector = torch.FloatTensor([0, 1, 0]).unsqueeze(0).repeat(len(centers), 1)
#     right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
#     up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

#     poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
#     poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
#     poses[:, :3, 3] = centers

#     return poses.numpy()

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

def gen_random_pos(size, param_range, gamma=1):
    lower, higher = param_range[0], param_range[1]
    
    mid = lower + (higher - lower) * 0.5
    radius = (higher - lower) * 0.5

    rand_ = torch.rand(size) # 0, 1
    sign = torch.where(torch.rand(size) > 0.5, torch.ones(size) * -1., torch.ones(size))
    rand_ = sign * (rand_ ** gamma)          

    return (rand_ * radius) + mid


def rand_poses(size, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5, rand_cam_gamma=1):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    # radius = torch.rand(size) * (radius_range[1] - radius_range[0]) + radius_range[0]
    radius = gen_random_pos(size, radius_range)

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size),
                torch.abs(torch.randn(size)),
                torch.randn(size),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        # thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
        # phis[phis < 0] += 2 * np.pi

        # centers = torch.stack([
        #     radius * torch.sin(thetas) * torch.sin(phis),
        #     radius * torch.cos(thetas),
        #     radius * torch.sin(thetas) * torch.cos(phis),
        # ], dim=-1) # [B, 3]
        # thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
        thetas = gen_random_pos(size, theta_range, rand_cam_gamma)
        phis = gen_random_pos(size, phi_range, rand_cam_gamma)
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.sin(thetas) * torch.cos(phis),
            radius * torch.cos(thetas),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center # 0.015  # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center/2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    #up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise) #forward_vector

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1) #up_vector
    poses[:, :3, 3] = centers


    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses.numpy(), thetas.numpy(), phis.numpy(), radius.numpy()

def GenerateMVAdapterCameras(opt):
    """Generate 6 cameras matching MV-Adapter's default view azimuths: [0, 45, 90, 180, 270, 315] degrees."""
    fov = opt.default_fovy
    cam_infos = []
    # MV-Adapter uses these 6 fixed azimuths
    mv_azimuths = [0, 45, 90, 180, 270, 315]
    for idx, az in enumerate(mv_azimuths):
        thetas = torch.FloatTensor([opt.default_polar])
        phis = torch.FloatTensor([az])
        radius = torch.FloatTensor([opt.default_radius])
        poses = circle_poses(radius=radius, theta=thetas, phi=phis,
                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                        width=opt.image_w, height=opt.image_h,
                                        delta_polar=delta_polar, delta_azimuth=delta_azimuth,
                                        delta_radius=delta_radius))
    return cam_infos

def GenerateCircleCameras(opt, size=8, render45 = False):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
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
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))  
    if render45:
        for idx in range(size):
            thetas = torch.FloatTensor([opt.default_polar*2//3])
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
                            height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))         
    return cam_infos

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
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))  
        
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
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))         
    return cam_infos

def GenerateRandomCameras(opt, size=2000, SSAA=True):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=opt.radius_range, theta_range=opt.theta_range, phi_range=opt.phi_range, 
                                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate,
                                             rand_cam_gamma=opt.rand_cam_gamma)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
    
    cam_infos = []

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    #generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # matrix = poses[idx]
        # R = matrix[:3,:3]
        # T = matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=image_w, 
                                        height=image_h, delta_polar = delta_polar[idx],
                                        delta_azimuth = delta_azimuth[idx], delta_radius = delta_radius[idx]))           
    return cam_infos

def GeneratePurnCameras(opt, size=300):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=[opt.default_radius,opt.default_radius+0.1], theta_range=opt.theta_range, phi_range=opt.phi_range, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    #fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])     
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # matrix = poses[idx]
        # R = matrix[:3,:3]
        # T = matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,width=opt.image_w, 
                        height = opt.image_h, delta_polar = delta_polar[idx],delta_azimuth = delta_azimuth[idx], delta_radius = delta_radius[idx]))           
    return cam_infos

def GenerateSoftPruneCameras(opt, size=16, render45 = False):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    #generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar, opt.default_polar, opt.default_polar, opt.default_polar,
                                    opt.default_polar+45, opt.default_polar+45, opt.default_polar+45, opt.default_polar+45,
                                    opt.default_polar-45, opt.default_polar-45, opt.default_polar-45, opt.default_polar-45,])
        phis = torch.FloatTensor([0, 90, 180, 270,
                                45, 135, 225, 315,
                                45, 135, 225, 315,])
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
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))  
    
    return cam_infos

def readVLMCamInfo(path,opt):
    test_cam_infos = GenerateOutCamerasVLM(opt)
    scene_info = RSceneInfo(point_cloud=None,
                           test_cameras=test_cam_infos,
                           ply_path='')
    return scene_info

def GenerateOutCamerasVLM(opt, size=8):
    # random focal
    fov = opt.default_fovy
    cam_infos = []

    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar-15])
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
                        height = opt.image_h, delta_polar = delta_polar,delta_azimuth = delta_azimuth, delta_radius = delta_radius))         
    return cam_infos

sceneLoadTypeCallbacks = {
    # "Colmap": readColmapSceneInfo,
    # "Blender" : readNerfSyntheticInfo,
    "RandomCam" : readCircleCamInfo,
    'VLMCam': readVLMCamInfo
}