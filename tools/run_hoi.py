# run HOI generation
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

import random
import imageio
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import math
from datetime import datetime
from torchvision.utils import save_image
import torchvision.transforms as T
from argparse import ArgumentParser, Namespace
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, matrix_to_axis_angle, rotation_6d_to_matrix
from pytorch3d.structures import Meshes

import yaml
import copy

import numpy as np
import json

import sys
stage1_dir = osp.join(osp.dirname(osp.abspath(osp.dirname(__file__))), 'GaussianDreamerPro', 'stage1')
sys.path.append(stage1_dir)

from utils.loss_utils import tv_loss,NormalLoss
from gaussian_renderer import render, network_gui
from utils.mano import mano
from scene import Scene, GaussianModel
from scene import HandScene, HandGaussianModel, HandGaussianWrapper
from scene import HOIGaussianModel, HOIScene
from utils.general_utils import safe_state
import uuid
from arguments import ModelParams, PipelineParams, OptimizationParams, GenerateCamParams, GuidanceParams, GenerateCamParamsHand, GenerateCamParamsHOI
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from utils.loss_utils import LaplacianReg, PenetrationLoss, GSScaleLoss
from utils.general_utils import get_expon_lr_func

def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    #TODO: add prenerg functions
    text_z_list = []
    weights_list = []
    K = 0
    #for b in range(azimuth):
    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt)
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def prepare_embeddings(guidance_opt, guidance, text, negative, inverse_text):
    embeddings = {}
    # text embeddings (stable-diffusion) and (IF)
    #embeddings['default'] = guidance.get_text_embeds([guidance_opt.text])
    text = guidance_opt.prefix_text + text + guidance_opt.postfix_text
    embeddings['default'] = guidance.get_text_embeds([text])
    embeddings['uncond'] = guidance.get_text_embeds([negative])

    for d in ['front', 'side', 'back']:
        #embeddings[d] = guidance.get_text_embeds([f"{guidance_opt.text}, {d} view"])
        embeddings[d] = guidance.get_text_embeds([f"{text}, {d} view"])
    #embeddings['inverse_text'] = guidance.get_text_embeds(guidance_opt.inverse_text)
    embeddings['inverse_text'] = guidance.get_text_embeds(inverse_text)
    return embeddings

def guidance_setup(guidance_opt):
    if guidance_opt.guidance=="SD":
        from guidance.sd_utils import StableDiffusion
        guidance = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                                   guidance_opt.t_range, guidance_opt.max_t_range, 
                                   num_train_timesteps=guidance_opt.num_train_timesteps, 
                                   ddim_inv=guidance_opt.ddim_inv,
                                   textual_inversion_path = guidance_opt.textual_inversion_path,
                                   LoRA_path = guidance_opt.LoRA_path,
                                   guidance_opt=guidance_opt)
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False
    #embeddings = prepare_embeddings(guidance_opt, guidance)
    negative = guidance_opt.negative
    obj_embeddings = prepare_embeddings(guidance_opt, guidance, guidance_opt.text, negative, guidance_opt.inverse_text)
    hand_embeddings = prepare_embeddings(guidance_opt, guidance, guidance_opt.hand_text, negative, guidance_opt.hand_inverse_text)
    hoi_embeddings = prepare_embeddings(guidance_opt, guidance, guidance_opt.hoi_text, negative, guidance_opt.hoi_inverse_text)
    return guidance, obj_embeddings, hand_embeddings, hoi_embeddings

def relax_view_range(scene, opt):
    scene.pose_args.fovy_range[0] = max(scene.pose_args.max_fovy_range[0], scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0])
    scene.pose_args.fovy_range[1] = min(scene.pose_args.max_fovy_range[1], scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1])

    scene.pose_args.radius_range[1] = max(scene.pose_args.max_radius_range[1], scene.pose_args.radius_range[1] * opt.scale_up_factor)
    scene.pose_args.radius_range[0] = max(scene.pose_args.max_radius_range[0], scene.pose_args.radius_range[0] * opt.scale_up_factor)

    scene.pose_args.theta_range[1] = min(scene.pose_args.max_theta_range[1], scene.pose_args.theta_range[1] * opt.phi_scale_up_factor)
    scene.pose_args.theta_range[0] = max(scene.pose_args.max_theta_range[0], scene.pose_args.theta_range[0] * 1/opt.phi_scale_up_factor)

    scene.pose_args.phi_range[0] = max(scene.pose_args.max_phi_range[0] , scene.pose_args.phi_range[0] * opt.phi_scale_up_factor)
    scene.pose_args.phi_range[1] = min(scene.pose_args.max_phi_range[1], scene.pose_args.phi_range[1] * opt.phi_scale_up_factor)

def batch_render(guidance_opt, embeddings, scene, gaussians, dataset, pipe, iteration, background, cam_type='train'):
    if cam_type == 'train':
        viewpoint_stack = scene.getRandTrainCameras().copy()         
    elif cam_type == 'clip':
        viewpoint_stack = scene.getCircleVideoCameras().copy()[120:]
        viewpoint_stack = viewpoint_stack[::15]
    elif cam_type == 'sd':
        viewpoint_stack = scene.getCircleVideoCameras().copy()[120:]
        viewpoint_stack = viewpoint_stack[::30]
    elif cam_type == 'sv3d':
        viewpoint_stack = scene.getCircleVideoCameras().copy()
        indices = np.linspace(0, len(viewpoint_stack) - 1, guidance_opt.C_batch_size, dtype=int)
        viewpoint_stack = [viewpoint_stack[i] for i in indices]

    
    C_batch_size = guidance_opt.C_batch_size
    viewpoint_cams = []
    images = []
    text_z_ = []
    weights_ = []
    depths = []
    alphas = []
    scales = []
    visibility_filter_list = []

    text_z_inverse =torch.cat([embeddings['uncond'],embeddings['inverse_text']], dim=0)

    for i in range(C_batch_size):
        try:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))            
        except:
            viewpoint_stack = scene.getRandTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
        #pred text_z
        azimuth = viewpoint_cam.delta_azimuth
        if isinstance(azimuth, torch.Tensor):
            azimuth = azimuth.cpu().numpy()[0]
        text_z = [embeddings['uncond']]


        if guidance_opt.perpneg:
            text_z_comp, weights = adjust_text_embeddings(embeddings, azimuth, guidance_opt)
            text_z.append(text_z_comp)
            weights_.append(weights)

        else:                
            if azimuth >= -90 and azimuth < 90:
                if azimuth >= 0:
                    r = 1 - azimuth / 90
                else:
                    r = 1 + azimuth / 90
                start_z = embeddings['front']
                end_z = embeddings['side']
            else:
                if azimuth >= 0:
                    r = 1 - (azimuth - 90) / 90
                else:
                    r = 1 + (azimuth + 90) / 90
                start_z = embeddings['side']
                end_z = embeddings['back']
            text_z.append(r * start_z + (1 - r) * end_z)
            

        text_z = torch.cat(text_z, dim=0)
        text_z_.append(text_z)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, 
                            sh_deg_aug_ratio = dataset.sh_deg_aug_ratio, 
                            bg_aug_ratio = dataset.bg_aug_ratio, 
                            shs_aug_ratio = dataset.shs_aug_ratio, 
                            scale_aug_ratio = dataset.scale_aug_ratio)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg["depth"]
        alpha = render_pkg["accum"]


        scales.append(render_pkg["scales"])
        images.append(image)
        depths.append(depth)
        alphas.append(alpha)
        viewpoint_cams.append(viewpoint_cams)
        visibility_filter_list.append(viewspace_point_tensor)


    images = torch.stack(images, dim=0)
    depths = torch.stack(depths, dim=0)
    alphas = torch.stack(alphas, dim=0)
    return_dict = { 
                    'viewpoint_cams': viewpoint_cams,
                    'images': images,
                    'text_z_': text_z_,
                    'weights_': weights_,
                    'depths': depths,
                    'alphas': alphas,
                    'scales': scales,
                    'visibility_filter_list': visibility_filter_list,
                    'text_z_inverse': text_z_inverse,
                    'render_pkg': render_pkg
                    }
    return return_dict

def get_loss(iteration, opt, guidance_opt, guidance, render_res, save_folder, gcams, Normal_Loss, 
            use_mano_mapping, gaussians, use_lap_loss, LapReg_Loss=None, GS_Scale_Loss=None):
    text_z_ = render_res['text_z_']
    images = render_res['images']
    alphas = render_res['alphas']
    text_z_inverse = render_res['text_z_inverse']
    depths = render_res['depths']
    scales = render_res['scales']
    render_pkg = render_res['render_pkg']


    warm_up_rate = 1. - min(iteration/opt.warmup_iter,1.)
    guidance_scale = guidance_opt.guidance_scale
    _aslatent = False
    if iteration < opt.geo_iter or random.random()< opt.as_latent_ratio:
        _aslatent=True
    use_control_net = False
    #if iteration > opt.use_control_net_iter and (random.random() < guidance_opt.controlnet_ratio):
        #use_control_net = True
    if guidance_opt.perpneg:
        loss = guidance.train_step_perpneg(torch.stack(text_z_, dim=1), images, 
                                            pred_depth=depths, pred_alpha=alphas,
                                            grad_scale=guidance_opt.lambda_guidance,
                                            use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                            weights = torch.stack(weights_, dim=1), resolution=(gcams.image_h, gcams.image_w),
                                            guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)
    else:
        loss = guidance.train_step(torch.stack(text_z_, dim=1), images, 
                                pred_depth=depths, pred_alpha=alphas,
                                grad_scale=guidance_opt.lambda_guidance,
                                use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                resolution=(gcams.image_h, gcams.image_w),
                                guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse)
        #raise ValueError(f'original version not supported.')
    scales = torch.stack(scales, dim=0)

    loss_scale = torch.mean(scales,dim=-1).mean()
    loss_tv = tv_loss(images) + tv_loss(depths)
    loss_bin = torch.mean(torch.min(alphas - 0.0001, 1 - alphas))

    loss = loss + opt.lambda_tv * loss_tv + \
        opt.lambda_scale * loss_scale  + opt.lambda_bin * loss_bin


    if use_lap_loss and LapReg_Loss is not None:
        loss_lap = gaussians.get_lap_loss(LapReg_Loss)
        loss = loss + loss_lap * opt.lambda_lap
    
    if opt.lambda_normal > 0:
        normal_loss = opt.lambda_normal * Normal_Loss(render_pkg["normal"], render_pkg["depth"], accums=render_pkg["accum"], stage=1)
        loss += normal_loss

    return loss


def get_hoi_params(hoi_gaussians, mano_param_dict, opt, dataset, penetration_loss, 
                    hoi_scene=None, render=None, pipe=None, background=None):
    obj_centric = dataset.obj_centric

    if not obj_centric: # optimize obj params
        raw_obj_rel_trans = torch.clone(hoi_gaussians.obj_rel_trans).detach()
        obj_rel_rot = torch.clone(matrix_to_axis_angle(hoi_gaussians.obj_rel_rot.detach())).requires_grad_(True)
        raw_obj_rel_rot = torch.clone(obj_rel_rot).detach()
        #hand_pose_aa = torch.clone(mano_param_dict()[0]['hand_pose'].detach()).requires_grad_(True)
        #raw_hand_hand_pose = torch.clone(hand_pose_aa).detach()
        #hoi_gaussians.hand_gaussians.hand_pose = hand_pose_aa
        root_pose_6d = mano_param_dict.mano_params['root_pose'].detach()
        raw_root_pose_6d = torch.clone(root_pose_6d).detach()
        hand_pose_6d = mano_param_dict.mano_params['hand_pose'].detach()
        raw_hand_pose_6d = torch.clone(hand_pose_6d).detach()
        #hoi_gaussians.hand_gaussians.hand_pose = nn.Parameter(hand_pose_6d.requires_grad_(False))

        # get min penetration translation
        move_vec = torch.zeros((1,3), dtype=raw_obj_rel_trans.dtype, device=raw_obj_rel_trans.device)

        hoi_param_dict = {
            'raw_obj_rel_trans': raw_obj_rel_trans,
            'raw_obj_rel_rot': raw_obj_rel_rot,
            #'raw_hand_hand_pose': raw_hand_hand_pose,
            'raw_hand_pose_6d': raw_hand_pose_6d,
            'obj_rel_rot': obj_rel_rot,
            'raw_root_pose_6d': raw_root_pose_6d,
            'root_pose_6d': root_pose_6d,
            #'hand_pose_aa': hand_pose_aa,
            #'hand_pose_6d': hoi_gaussians.hand_gaussians.hand_pose,
            'hand_pose_6d': hand_pose_6d,
            'move_vec': move_vec
        }
    else:
        raw_hand_rel_trans = torch.clone(hoi_gaussians.hand_rel_trans).detach()
        obj_rel_rot = torch.clone(matrix_to_axis_angle(hoi_gaussians.obj_rel_rot.detach())).requires_grad_(True)
        raw_obj_rel_rot = torch.clone(obj_rel_rot).detach()
        #hand_rel_rot = torch.clone(matrix_to_axis_angle(hoi_gaussians.hand_rel_rot.detach())).requires_grad_(True)
        #raw_hand_rel_rot = torch.clone(hand_rel_rot).detach()

        root_pose_6d = mano_param_dict.mano_params['root_pose'].detach()
        raw_root_pose_6d = torch.clone(root_pose_6d).detach()
        hand_pose_6d = mano_param_dict.mano_params['hand_pose'].detach()
        raw_hand_pose_6d = torch.clone(hand_pose_6d).detach()
        
        if opt.get_closer:
            hand_xyz = hoi_gaussians.hand_gaussians.get_xyz
            obj_xyz = hoi_gaussians.obj_gaussians.get_xyz
        else:
            move_vec = torch.zeros((1,3), dtype=raw_hand_rel_trans.dtype, device=raw_hand_rel_trans.device)
        hoi_param_dict = {
            'raw_hand_rel_trans': raw_hand_rel_trans,
            #'raw_hand_rel_rot': raw_hand_rel_rot,
            'raw_obj_rel_rot': raw_obj_rel_rot,
            'raw_hand_pose_6d': raw_hand_pose_6d,
            'raw_root_pose_6d': raw_root_pose_6d,
            #'hand_rel_rot': hand_rel_rot,
            'obj_rel_rot': obj_rel_rot,
            'root_pose_6d': root_pose_6d,
            'hand_pose_6d': hand_pose_6d,
            'move_vec': move_vec
        }
    return hoi_param_dict

def get_hoi_opt_list(opt, dataset, hoi_gaussians, hoi_param_dict):
    #hand_pose_aa = hoi_param_dict['hand_pose_aa']
    #hoi_param_dict['hand_pose_6d'].requires_grad = True
    #hand_pose_6d = hoi_param_dict['hand_pose_6d']
    hoi_gaussians.hand_gaussians.root_pose = nn.Parameter(hoi_param_dict['root_pose_6d'].requires_grad_(True))
    hoi_gaussians.hand_gaussians.hand_pose = nn.Parameter(hoi_param_dict['hand_pose_6d'].requires_grad_(True))
    global_lr_scale = opt.global_lr_scale

    global_param_list = [
        {'params': [hoi_gaussians.hand_gaussians.hand_pose], 'lr': opt.rotation_lr * global_lr_scale, 'name': 'hand_hand_pose'},
        {'params': [hoi_gaussians.hand_gaussians.root_pose], 'lr': opt.rotation_lr * global_lr_scale, 'name': 'hand_root_pose'},
    ]
    if dataset.obj_centric:
        global_param_list += [
            {'params': [hoi_gaussians.hand_rel_trans], 'lr': opt.rotation_lr * global_lr_scale, 'name': 'hand_rel_trans'}
        ]
    else:
        #obj_rel_rot = hoi_param_dict['obj_rel_rot']
        global_param_list += [
            {'params': [hoi_gaussians.obj_rel_trans], 'lr': opt.rotation_lr * global_lr_scale, 'name': 'obj_rel_trans'}
        ]
    return global_param_list

def get_hoi_expon_lr_func(opt):
    global_lr_scale = opt.global_lr_scale
    pos_scheduler_args = get_expon_lr_func(lr_init=opt.rotation_lr * global_lr_scale,
                                            lr_final=opt.rotation_lr_final,
                                            lr_delay_mult=opt.position_lr_delay_mult,
                                            max_steps=opt.position_lr_max_steps)
    rot_scheduler_args = get_expon_lr_func(lr_init=opt.rotation_lr * global_lr_scale,
                                            lr_final=opt.rotation_lr_final,
                                            lr_delay_mult=opt.position_lr_delay_mult,
                                            max_steps=opt.iterations)
    scaling_scheduler_args = get_expon_lr_func(lr_init=opt.scaling_lr * global_lr_scale,
                                            lr_final=opt.scaling_lr_final,
                                            lr_delay_mult=opt.position_lr_delay_mult,
                                            max_steps=opt.iterations)
    return pos_scheduler_args, rot_scheduler_args, scaling_scheduler_args

def update_hoi_lr(global_optimizer, iteration, pos_scheduler_args, rot_scheduler_args, scaling_scheduler_args):
    for param_group in global_optimizer.param_groups:
        if param_group['name'] in ['obj_rel_trans', 'hand_rel_trans','hand_trans']:
            lr = pos_scheduler_args(iteration)
            param_group['lr'] = lr
        elif param_group['name'] in ['obj_rel_rot', 'hand_root_pose', 'hand_hand_pose', 'obj_rot', 'hand_rot']:
            lr = rot_scheduler_args(iteration)
            param_group['lr'] = lr
        elif param_group['name'] in ['obj_rescaling']:
            lr = scaling_scheduler_args(iteration)
            param_group['lr'] = lr

def switch_hoi_param_training(hoi_gaussians, obj_rel_rot, switch):
    obj_gaussians = hoi_gaussians.obj_gaussians
    hand_gaussians = hoi_gaussians.hand_gaussians
    if switch == 'freeze':
        # freeze hoi params
        if obj_rel_rot is not None:
            obj_rel_rot.requires_grad = False
        hoi_gaussians.obj_rel_trans.requires_grad = False
        hoi_gaussians.hand_gaussians.root_pose.requires_grad = False
        hoi_gaussians.hand_gaussians.hand_pose.requires_grad = False

    else:
        # thaw hoi params
        if obj_rel_rot is not None:
            obj_rel_rot.requires_grad = True
        hoi_gaussians.obj_rel_trans.requires_grad = True
        hoi_gaussians.hand_gaussians.root_pose.requires_grad = True
        hoi_gaussians.hand_gaussians.hand_pose.requires_grad = True



def training_hoi(dataset, opt, pipe, gcams_obj, gcams_hand, gcams_hoi, 
                guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, 
                debug_from, save_video, gpu_id):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # object gaussian model initialization
    gaussian_cfg = AttrDict({
                        'isotropic': opt.isotropic
                        })
    obj_gaussians = GaussianModel(dataset.sh_degree, gaussian_cfg)

    # hand gauassian model initialization
    shape_param = torch.zeros((1,10), dtype=torch.float32, device='cuda')
    mano.set_id_info(shape_param)

    use_mano_mapping = dataset.use_mano_mapping
    use_triplane = dataset.use_triplane
    learn_lbs_offset = dataset.learn_lbs_offset
    hand_subdivide_num = dataset.hand_subdivide_num
    hand_color_init = opt.hand_color_init
    hand_gaussian_cfg = AttrDict({
                        'mano_rhand': True,
                        'hand_subdivide_num': hand_subdivide_num,
                        'learn_mano_param': opt.learn_mano_param,
                        'learn_lbs_offset': learn_lbs_offset,
                        'isotropic': opt.isotropic,
                        'hand_rescaling': opt.hand_rescaling,
                        'hand_color_init': hand_color_init
                        })
    if use_mano_mapping:
        if use_triplane:
            raise NotImplementedError
        else:
            hand_gaussians = HandGaussianWrapper(dataset.sh_degree, hand_gaussian_cfg)
    else:
        hand_gaussians = HandGaussianModel(dataset.sh_degree, hand_gaussian_cfg)
    
    obj_dataset = copy.deepcopy(dataset)
    if dataset.pretrained_obj_ply is not None:
        obj_dataset.pretrained_model_path = dataset.pretrained_obj_ply
    obj_scene = Scene(obj_dataset, gcams_obj, obj_gaussians)
    
    hand_dataset = copy.deepcopy(dataset)
    if dataset.pretrained_hand_ply is not None:
        hand_dataset.pretrained_model_path = dataset.pretrained_hand_ply
    hand_scene = HandScene(hand_dataset, gcams_hand, hand_gaussians)

    hoi_gaussians = HOIGaussianModel(dataset.sh_degree, obj_gaussians, hand_gaussians)
    hoi_scene = HOIScene(dataset, gcams_hoi, hoi_gaussians)
    
    if dataset.obj_ref_ckpt != '' and osp.exists(dataset.obj_ref_ckpt):
        print(f'loading pretrained object gaussian model from {dataset.obj_ref_ckpt}')
        res = torch.load(dataset.obj_ref_ckpt, weights_only=False)
        (obj_model_params, first_iter_obj_ref) = res
        obj_gaussians.restore(obj_model_params, None)

    if dataset.hand_ref_ckpt != '' and osp.exists(dataset.hand_ref_ckpt):
        print(f'loading pretrained hand gaussian model from {dataset.hand_ref_ckpt}')
        hand_res = torch.load(dataset.hand_ref_ckpt, weights_only=False)
        (hand_model_params, first_iter_hand_ref) = hand_res
        hand_gaussians.restore(hand_model_params, None)


    hoi_gaussians.training_setup(opt, hand_scene.mano_param_dict)

    if checkpoint:
        (obj_model_params, first_iter) = torch.load(checkpoint.replace('.pth', '_obj.pth'), weights_only=False)
        obj_gaussians.restore(obj_model_params, opt)

        (hand_model_params, first_iter) = torch.load(checkpoint.replace('.pth', '_hand.pth'), weights_only=False)
        hand_gaussians.restore(hand_model_params, opt)

        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        hoi_gaussians.restore(model_params, opt)

        json_path = checkpoint.replace('.pth', '_mano_params.json')
        if os.path.exists(json_path):
            print(f'loading fitted hand pose parameters: {json_path}')
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            root_pose = torch.tensor(json_data['root_pose'])
            if isinstance(hoi_gaussians.hand_gaussians.root_pose, nn.Parameter):
                requires_grad = hoi_gaussians.hand_gaussians.root_pose._requires_grad
                root_pose = nn.Parameter(root_pose, requires_grad=requires_grad)
            hoi_gaussians.hand_gaussians.root_pose = root_pose

            hoi_gaussians.hand_gaussians.hand_pose = torch.tensor(json_data['hand_pose'])
            if isinstance(hoi_gaussians.hand_gaussians.hand_pose, nn.Parameter):
                requires_grad = hoi_gaussians.hand_gaussians.hand_pose._requires_grad
                hand_pose = nn.Parameter(hand_pose, requires_grad=requires_grad)
            hoi_gaussians.hand_gaussians.hand_pose = hand_pose
    
    # freeze xyz
    obj_gaussians._xyz.requires_grad = False
    if not opt.joint_hand_hoi_opt:
        hand_gaussians.mean_offset.requires_grad = False
        if hand_gaussians.learn_lbs_offset:
            hand_gaussians.mean_offset_offset.requires_grad = False

    # freeze color offset
    hand_gaussians.f_dc_offset.requires_grad = False
    hand_gaussians.f_rest_offset.requires_grad = False
    
    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    save_folder = os.path.join(dataset._model_path,"train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs

    obj_save_folder = os.path.join(dataset._model_path,"train_process_obj/")
    os.makedirs(obj_save_folder, exist_ok=True)
    hand_save_folder = os.path.join(dataset._model_path,"train_process_hand/")
    os.makedirs(hand_save_folder, exist_ok=True)
    use_control_net = False
    guidance, obj_embeddings, hand_embeddings, hoi_embeddings = guidance_setup(guidance_opt)   

    if opt.opt_with_coarse:
        if opt.use_undersample:
            obj_coarse_ply_path = os.path.join(*dataset.pretrained_obj_ply.split(os.sep)[:-3], 'meshify_5000_subsample.ply')
        else:
            #obj_coarse_ply_path = dataset.pretrained_obj_ply.replace('.ply', '_coarse.ply')
            obj_coarse_ply_path = os.path.join(*dataset._model_path.split(os.sep)[:-1], 'obj', 'meshify_5000_coarse.ply')
        obj_gaussians.load_coarse_mesh(obj_coarse_ply_path)

    use_penetration_loss = opt.use_penetration_loss
    penet_loss_lr = opt.penetration_lr
    if use_penetration_loss:
        penetration_loss = PenetrationLoss(obj_gaussians, hand_gaussians, 
                                        opt.use_joint_contact, opt.use_obj_contact,
                                        opt.use_penet_sum, opt.use_erf,
                                        opt.opt_with_coarse,
                                        opt.use_ik_loss)
        penetration_loss.get_hand_contact_joints(hoi_gaussians)
        penetration_loss.get_obj_contact_verts(hoi_gaussians)
    else:
        penetration_loss = None
    
    # global optimizer
    use_global_opt = opt.use_global_opt
    global_opt_method = opt.global_opt_method
    joint_global_opt = global_opt_method == 'joint'

    hoi_guidance_opt = copy.deepcopy(guidance_opt)
    #guidance_opt.C_batch_size = max(guidance_opt.C_batch_size * 3, 12) # increase views

    hoi_guidance_opt.C_batch_size = 4

    viewpoint_stack = None
    #viewpoint_stack_around = None
    ema_loss_for_log = 0.0

    if opt.lambda_normal > 0:
        viewpoint_camera = hoi_scene.getRandTrainCameras()[0]
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
        focal_x = W / (2 * tanfovx)
        focal_y = H / (2 * tanfovy)
        Normal_Loss = NormalLoss(H, W, focal_x, focal_y, device="cuda")

    GS_Scale_Loss = GSScaleLoss()

    use_lap_loss = opt.lambda_lap > 0
    LapReg_loss_obj = None
    LapReg_loss_hand = None
    if use_lap_loss:
        # object lap loss
        obj_ply_path = dataset.pretrained_obj_ply
        obj_gaussians.load_mesh(obj_ply_path)
        if hasattr(obj_gaussians, 'faces'):
            obj_n_v = obj_gaussians._xyz.shape[0]
            obj_faces = obj_gaussians.faces.detach().cpu().numpy()
            LapReg_loss_obj = LaplacianReg(obj_n_v, obj_faces)

        # hand lap loss
        LapReg_loss_hand = LaplacianReg(mano.vertex_num_upsampled, mano.face_upsampled)

    if opt.save_process:
        save_folder_proc = os.path.join(hoi_scene.args._model_path,"process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)  # makedirs
        process_view_points = hoi_scene.getCircleVideoCameras(batch_size=opt.pro_frames_num,render45=opt.pro_render_45).copy()    
        save_process_iter = opt.iterations // len(process_view_points)
        pro_img_frames = []
    
    
    obj_centric = dataset.obj_centric
    if use_global_opt:
        if joint_global_opt:
            global_lr_scale = opt.global_lr_scale
            hoi_param_dict = get_hoi_params(hoi_gaussians, hand_scene.mano_param_dict, opt, dataset, penetration_loss)

            if obj_centric:
                raw_hand_rel_trans = hoi_param_dict['raw_hand_rel_trans']
            else:
                raw_obj_rel_trans = hoi_param_dict['raw_obj_rel_trans']
            #raw_obj_rel_rot = hoi_param_dict['raw_obj_rel_rot']
            raw_root_pose_6d = hoi_param_dict['raw_root_pose_6d']
            root_pose_6d = hoi_param_dict['root_pose_6d']
            raw_hand_pose_6d = hoi_param_dict['raw_hand_pose_6d']
            #obj_rel_rot = hoi_param_dict['obj_rel_rot']
            #hand_pose_aa = hoi_param_dict['hand_pose_aa']
            hand_pose_6d = hoi_param_dict['hand_pose_6d']
            move_vec = hoi_param_dict['move_vec']

            use_best = False
            #best_obj_rel_rot = torch.clone(obj_rel_rot.detach()).requires_grad_(False)
            best_obj_rel_trans = torch.clone(hoi_gaussians.obj_rel_trans.detach()).requires_grad_(False)
            best_root_pose_6d = torch.clone(root_pose_6d.detach()).requires_grad_(False)
            best_hand_pose_6d = torch.clone(hand_pose_6d.detach()).requires_grad_(False)
            best_loss = 9000.0
            best_it = -1
            
            # get optimizer
            global_param_list = get_hoi_opt_list(opt, dataset, hoi_gaussians, hoi_param_dict)
            switch_hoi_param_training(hoi_gaussians, None, 'thaw')
            global_optimizer = torch.optim.Adam(global_param_list, lr=0.0, eps=1e-15)

            pos_scheduler_args, rot_scheduler_args, scaling_scheduler_args = get_hoi_expon_lr_func(opt)

            # get contact embeddings
            obj_text = guidance_opt.text
            hand_text = guidance_opt.hand_text
            hoi_text = guidance_opt.hoi_text
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        hoi_gaussians.update_learning_rate(iteration)
        hoi_gaussians.update_feature_learning_rate(iteration)
        hoi_gaussians.update_rotation_learning_rate(iteration)
        hoi_gaussians.update_scaling_learning_rate(iteration)
        # Every 500 its we increase the levels of SH up to a maximum degree
        """
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        """
        # progressively relaxing view range    
        if not opt.use_progressive:                
            if iteration >= opt.progressive_view_iter and iteration % opt.scale_up_cameras_iter == 0:
                relax_view_range(obj_scene, opt)
                relax_view_range(hand_scene, opt)
                relax_view_range(hoi_scene, opt)

        obj_render_res = batch_render(guidance_opt, obj_embeddings, obj_scene, obj_gaussians, dataset, pipe, iteration, background)
        hand_render_res = batch_render(guidance_opt, hand_embeddings, hand_scene, hand_gaussians, dataset, pipe, iteration, background)
        #hoi_render_res = batch_render(guidance_opt, hoi_embeddings, hoi_scene, hoi_gaussians, dataset, pipe, iteration, background)

        obj_loss = .0
        obj_loss = get_loss(iteration, opt, guidance_opt, guidance, obj_render_res, obj_save_folder, gcams_obj, Normal_Loss, use_mano_mapping, obj_gaussians, use_lap_loss, LapReg_loss_obj, GS_Scale_Loss)
        hand_loss = get_loss(iteration, opt, guidance_opt, guidance, hand_render_res, hand_save_folder, gcams_hand, Normal_Loss, use_mano_mapping, hand_gaussians, use_lap_loss, LapReg_loss_hand, GS_Scale_Loss)
        #hoi_loss = .0
        #hoi_loss = get_loss(iteration, opt, guidance_opt, guidance, hoi_render_res, save_folder, gcams_hoi, Normal_Loss, use_mano_mapping, hoi_gaussians, use_lap_loss, None)

        loss = obj_loss + hand_loss

        #loss.backward(retain_graph=True)
        loss.backward()
        iter_end.record()

        with torch.no_grad(): # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(viewpoint_cam_p, hoi_gaussians, pipe, background, test=True)
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0)
                    img_p = img_p.detach().cpu().permute(1,2,0).numpy()
                    img_p = (img_p * 255).round().astype('uint8')
                    pro_img_frames.append(img_p)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{3}f}",
                                          "gs number": f"{hoi_gaussians.get_opacity.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, iter_start.elapsed_time(iter_end), testing_iterations, hoi_scene, render, (pipe, background))
            if (iteration in testing_iterations):
                if save_video:
                    video_inference(iteration, hoi_scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                obj_scene.save(iteration)
                hand_scene.save(iteration)
                hoi_scene.save(iteration)
            
            # Optimizer step
            if iteration < opt.iterations:
                hoi_gaussians.optimizer.step()
                hoi_gaussians.optimizer.zero_grad(set_to_none = True)
            
            # opacity control
            if iteration % opt.densification_interval == 0:
                hoi_gaussians.clamp_obj_opacity(opt.min_opacity)
        
        if use_global_opt and joint_global_opt and iteration < opt.iterations:
            # thaw hoi params
            switch_hoi_param_training(hoi_gaussians, None, 'thaw')

            update_hoi_lr(global_optimizer, iteration, pos_scheduler_args, rot_scheduler_args, scaling_scheduler_args)

            # parameter consistency
            if obj_centric:
                trans_cons = 20.0 * F.mse_loss(hoi_gaussians.hand_rel_trans, raw_hand_rel_trans)
            else:
                trans_cons = 20.0 * F.mse_loss(hoi_gaussians.obj_rel_trans, raw_obj_rel_trans)
            #rot_cons = F.mse_loss(obj_rel_rot, raw_obj_rel_rot)/raw_obj_rel_rot.max()
            #rescale_cons = 100 * F.mse_loss(hoi_gaussians.obj_rescaling, raw_obj_rescaling)/raw_obj_rescaling.max()
            #hand_pose_cons = F.mse_loss(mano_param_dict.mano_params['hand_pose'], raw_hand_hand_pose)
            root_pose_cons = F.mse_loss(hoi_gaussians.hand_gaussians.root_pose, raw_root_pose_6d)
            hand_pose_cons = F.mse_loss(hoi_gaussians.hand_gaussians.hand_pose, raw_hand_pose_6d)
            cons_loss = trans_cons + hand_pose_cons + root_pose_cons

            xy_axis_mask = torch.arange(3, device=raw_hand_pose_6d.device)%3!=2
            xy_axis_mask = xy_axis_mask.reshape(1,-1).repeat(15,1)
            hand_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(hoi_gaussians.hand_gaussians.hand_pose))
            raw_hand_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(raw_hand_pose_6d))
            axis_reg_loss = F.mse_loss(hand_pose_aa, raw_hand_pose_aa, reduction='none')
            axis_reg_loss = 10.0 * torch.sum(axis_reg_loss * xy_axis_mask)
            hoi_loss_msg = f'cons: {cons_loss:.3e}, axis_reg: {axis_reg_loss:.3e}'
            if penetration_loss is not None:
                penet_loss_dict = penetration_loss(hoi_gaussians,
                                                            pene_weight=opt.pene_weight,
                                                            contact_loss_weight=opt.hand_contact_weight,
                                                            obj_contact_weight=opt.obj_contact_weight,
                                                            ik_loss_weight=opt.ik_loss_weight,
                                                            return_dict=True)
                penet_loss_sum = .0
                for k in penet_loss_dict.keys():
                    penet_loss_sum = penet_loss_sum + penet_loss_dict[k]
                    hoi_loss_msg += f', {k}: {penet_loss_dict[k]:.3e}'
                penet_loss = opt.penetration_lr * penet_loss_sum
            else:
                penet_loss = torch.zeros(1).cuda().sum()
            
            if iteration % 100 == 0:
                print(hoi_loss_msg)
            
            hoi_opt_loss = penet_loss

            if opt.use_cons_loss:
                hoi_opt_loss = hoi_opt_loss + opt.lambda_hoi_cons * cons_loss + axis_reg_loss
            
            if (penet_loss.item() == 0.0) and not opt.use_cons_loss:
                continue
            
            hoi_opt_loss.backward()
            global_optimizer.step()
            #if iteration % 10 == 0:
                #print(f'after backward and step. hoi_gaussians.hand_gaussians.hand_pose[0]: {hoi_gaussians.hand_gaussians.hand_pose[0]} hoi_gaussians.hand_gaussians.hand_pose.grad[0] after backward: {hoi_gaussians.hand_gaussians.hand_pose.grad[0]}')
            global_optimizer.zero_grad(set_to_none = True)
            hoi_gaussians.optimizer.zero_grad(set_to_none = True)

            # clamp pose
            with torch.no_grad():
                hoi_gaussians.hand_gaussians.root_pose.data = torch.clamp(hoi_gaussians.hand_gaussians.root_pose.data, -3.14, 3.14)
                hoi_gaussians.hand_gaussians.hand_pose.data = torch.clamp(hoi_gaussians.hand_gaussians.hand_pose.data, -0.6, 1.65)
            #hoi_gaussians.hand_gaussians.root_pose.data = torch.clamp(hoi_gaussians.hand_gaussians.root_pose.data, -1.0, 1.0)
            #hoi_gaussians.hand_gaussians.hand_pose.data = torch.clamp(hoi_gaussians.hand_gaussians.hand_pose.data, -1.0, 1.0)

            # freeze hoi params
            switch_hoi_param_training(hoi_gaussians, None, 'freeze')
        
        with torch.no_grad():
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((obj_gaussians.capture(), iteration), hoi_scene.model_path + "/chkpnt" + str(iteration) + "_obj.pth")
                torch.save((hand_gaussians.capture(), iteration), hoi_scene.model_path + "/chkpnt" + str(iteration) + "_hand.pth")
                torch.save((hoi_gaussians.capture(), iteration), hoi_scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                # save mano pose
                json_path = hoi_scene.model_path + "/chkpnt" + str(iteration) + "_mano_params.json"
                with open(json_path, 'w') as f:
                    mano_save_dict = {
                        'root_pose': hoi_gaussians.hand_gaussians.root_pose.detach().cpu().numpy().tolist(),
                        'hand_pose': hoi_gaussians.hand_gaussians.hand_pose.detach().cpu().numpy().tolist(),
                    }
                    json.dump(mano_save_dict, f)
    if opt.save_process:
        imageio.mimwrite(os.path.join(save_folder_proc, "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)

    return dataset._model_path


def prepare_output_and_logger(args):    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not args._model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args._model_path = os.path.join("./output/", args.workspace+ '@'+ timestamp)
        
    # Set up output folder
    print("Output folder: {}".format(args._model_path))
    os.makedirs(args._model_path, exist_ok = True)

    # copy configs
    if args.opt_path is not None:
        os.system(' '.join(['cp', args.opt_path, os.path.join(args._model_path, 'config.yaml')]))

    with open(os.path.join(args._model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    """
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args._model_path)
    else:
        print("Tensorboard not available: not logging progress")
    """
    return tb_writer

def training_report(tb_writer, iteration, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        save_folder = os.path.join(scene.args._model_path,"test_six_views/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print('test views is in :', save_folder)
        torch.cuda.empty_cache()
        config = ({'name': 'test', 'cameras' : scene.getTestCameras()})
        if config['cameras'] and len(config['cameras']) > 0:
            for idx, viewpoint in enumerate(config['cameras']):
                render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
                rgb, depth = render_out["render"],render_out["depth"]
                if depth is not None:
                    depth_norm = depth/depth.max()
                    save_image(depth_norm,os.path.join(save_folder,"render_depth_{}.png".format(viewpoint.uid)))
                normal = render_out["normal"]
                save_image(normal,os.path.join(save_folder,"render_normal_{}.png".format(viewpoint.uid)))
                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image,os.path.join(save_folder,"render_view_{}.png".format(viewpoint.uid)))
                if tb_writer:
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.uid), image[None], global_step=iteration)     
            print("\n[ITER {}] Eval Done!".format(iteration))
        #if tb_writer:
            #tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            #tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            #tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def video_inference(iteration, scene : Scene, renderFunc, renderArgs):
    sharp = T.RandomAdjustSharpness(3, p=1.0)

    save_folder = os.path.join(scene.args._model_path,"videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs 
        print('videos is in :', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras' : scene.getCircleVideoCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        depth_frames = []
        #print("Generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb,depth = render_out["render"],render_out["depth"]
            if depth is not None:
                depth_norm = depth/depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0) 
                depths = depths.detach().cpu().permute(1,2,0).numpy()
                depths = (depths * 255).round().astype('uint8')          
                depth_frames.append(depths)    
  
            image = torch.clamp(rgb, 0.0, 1.0) 
            image = image.detach().cpu().permute(1,2,0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)    
            #save_image(image,os.path.join(save_folder,"lora_view_{}.jpg".format(viewpoint.uid)))   
        # Img to Numpy
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, quality=8)
        if len(depth_frames) > 0:
            imageio.mimwrite(os.path.join(save_folder, "video_depth_{}.mp4".format(iteration)), depth_frames, fps=30, quality=8)
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()


class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_ratio", type=int, default=2) # [2500,5000,7500,10000,12000]
    parser.add_argument("--save_ratio", type=int, default=2) # [10000,12000]
    parser.add_argument("--save_video", type=bool, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--obj_prompt", type=str, default = '')
    parser.add_argument("--obj_initprompt", type=str, default = None)
    parser.add_argument("--hand_prompt", type=str, default = '')
    parser.add_argument("--hoi_prompt", type=str, default = '')
    parser.add_argument("--t2hoi_prompt", type=str, default = '')
    parser.add_argument("--tag", type=str, default = None)
    parser.add_argument("--text2hoi_pkl", type=str, default = '')
    parser.add_argument("--obj_pth", type=str, default = '')
    parser.add_argument("--obj_ply", type=str, default = '')
    parser.add_argument("--hand_pth", type=str, default = '')
    parser.add_argument("--hand_ply", type=str, default = '')
    parser.add_argument("--output_dir", type=str, default = '')
    parser.add_argument("--gpu_id", type=int, default = 0)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gcp_obj = GenerateCamParams(parser)
    gp = GuidanceParams(parser)
    #gcp_hand = GenerateCamParamsHand(parser)
    #gcp_hoi = GenerateCamParamsHOI(parser)
    gcp_hand = copy.deepcopy(gcp_obj)
    gcp_hoi = copy.deepcopy(gcp_obj)

    args = parser.parse_args(sys.argv[1:])

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        lp.load_yaml(opts.get('ModelParams', None))
        op.load_yaml(opts.get('OptimizationParams', None))
        pp.load_yaml(opts.get('PipelineParams', None))
        gcp_obj.load_yaml(opts.get('GenerateCamParams', None))
        gcp_hand.load_yaml(opts.get('GenerateCamParamsHand', None))
        gcp_hoi.load_yaml(opts.get('GenerateCamParamsHOI', None))
        gp.load_yaml(opts.get('GuidanceParams', None))
        
        lp.opt_path = args.opt
        args.port = opts['port']
        args.save_video = opts.get('save_video', True)
        args.seed = opts.get('seed', 0)
        args.device = opts.get('device', 'cuda')

        # override device
        gp.g_device = args.device
        lp.data_device = args.device
        gcp_obj.device = args.device
        gcp_hand.device = args.device
        gcp_hoi.device = args.device
    
    if args.output_dir != '':
        lp._model_path = args.output_dir
    
    if args.obj_prompt != '':
        gp.text = args.obj_prompt
    if args.obj_initprompt is not None:
        gcp_obj.init_prompt = args.obj_initprompt
    elif gcp_obj.init_prompt == '':
        gcp_obj.init_prompt = gp.text

    if args.hand_prompt != '':
        gp.hand_text = args.hand_prompt
    if args.hoi_prompt != '':
        gp.hoi_text = args.hoi_prompt
    if args.hoi_prompt != '':
        gp.t2hoi_text = args.t2hoi_prompt
    else:
        gp.t2hoi_text = args.hoi_prompt
    device_idx = 0
        
    #lp.workspace = args.prompt.replace(' ', '_')
    #lp.workspace = args.initprompt.replace(' ', '_')
    lp.workspace = gp.hoi_text.split(',')[0].replace(' ', '_')

    # save iterations
    #test_iter = [1] + [k * op.iterations // args.test_ratio for k in range(1, args.test_ratio)] + [op.iterations]
    test_iter = [1, op.iterations]
    args.test_iterations = test_iter

    #save_iter = [k * op.iterations // args.save_ratio for k in range(1, args.save_ratio)] + [op.iterations]
    save_iter = [1, op.iterations]
    args.save_iterations = save_iter
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations = save_iter
    
    if args.text2hoi_pkl != '':
        lp.mano_path = args.text2hoi_pkl

    if args.obj_pth != '':
        lp.obj_ref_ckpt = args.obj_pth
    
    if args.obj_ply != '':
        lp.pretrained_obj_ply = args.obj_ply
    
    if args.hand_ply != '':
        lp.pretrained_hand_ply = args.hand_ply
    
    if args.hand_pth != '':
        lp.hand_ref_ckpt = args.hand_pth

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    start_time = datetime.now()

    checkpoint_dir = training_hoi(lp, op, pp, gcp_obj, gcp_hand, gcp_hoi, 
                            gp, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
                            args.debug_from, args.save_video, args.gpu_id)

    end_time = datetime.now()

    print("hoi training complete.\n")
    print(f'checkpoint_dir: {checkpoint_dir}')
    elapsed_time = end_time - start_time
    print("elapsed time: ", elapsed_time)
    
    
    