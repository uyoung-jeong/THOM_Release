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

import torch
import numpy as np
from torch import nn
import os
from plyfile import PlyData, PlyElement
import copy
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.mano import mano
from utils.smplx.smplx.lbs import batch_rigid_transform
from .hand_gaussian_model import HandGaussianModel


class HandGaussianWrapper(HandGaussianModel):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, cfg):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.isotropic = cfg.isotropic
        self.learn_mano_param = cfg.learn_mano_param
        self.hand_color_init = cfg.hand_color_init

        self.hand_subdivide_num = cfg.hand_subdivide_num
        mano.assign_new_subdivide_num(self.hand_subdivide_num)

        self.mano_layer = copy.deepcopy(mano.layer[cfg.mano_rhand]).cuda()
        self.shape_param = nn.Parameter(mano.shape_param.float().cuda())
        #self.shape_param = nn.Parameter(torch.zeros((1,10), dtype=torch.float32, device='cuda'))

        # exavatar style, but without triplane
        self.mean_offset = torch.empty(0) # offset for template pose
        self.scale_offset = torch.empty(0) # offset for articulated pose

        self.learn_lbs_offset = cfg.learn_lbs_offset
        self.mean_offset_offset = torch.empty(0) # offset for articulated pose. this is not used since we directly use mano lbs weight

        #self.rgb_offset = torch.empty(0) # offset for articulated pose
        self.f_dc_offset = torch.empty(0)
        self.f_rest_offset = torch.empty(0)

        # upsample mesh and other assets
        self.hand_trans = None
        self.hand_rescaling = cfg.hand_rescaling
        xyz_neutral, _, _, _ = self.get_neutral_pose_hand(use_id_info=False)
        skinning_weight = self.mano_layer.lbs_weights.float()
        pose_dirs = self.mano_layer.posedirs.permute(1,0).reshape(mano.vertex_num,3*(mano.joint_num-1)*9)
        
        _, skinning_weight, pose_dirs = mano.upsample_mesh(torch.ones((mano.vertex_num,3)).float().cuda(), [skinning_weight, pose_dirs]) # upsample with dummy vertex

        pose_dirs = pose_dirs.reshape(mano.vertex_num_upsampled*3,(mano.joint_num-1)*9).permute(1,0) 
        """
        self.register_buffer('pos_enc_mesh', xyz_neutral)
        self.register_buffer('skinning_weight', skinning_weight)
        self.register_buffer('pose_dirs', pose_dirs)
        """
        self.pos_enc_mesh = xyz_neutral
        self.skinning_weight = skinning_weight
        self.pose_dirs = pose_dirs

        self.root_pose = torch.empty(0)
        self.hand_pose = torch.empty(0) # use 6d representation
        self.trans = torch.empty(0)

    def get_neutral_pose_hand(self, use_id_info):
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_hand_pose = mano.neutral_hand_pose.view(1,-1).cuda() # tempalte pose

        if use_id_info:
            shape_param = self.shape_param
            if shape_param.dim()==1:
                shape_param = shape_param[None,:]
        else:
            shape_param = torch.zeros_like(mano.shape_param).float().cuda()
        
        #output = self.mano_layer(global_orient=zero_pose, hand_pose=neutral_hand_pose, betas=shape_param, transl=self.hand_trans)
        output = self.mano_layer(global_orient=zero_pose, hand_pose=neutral_hand_pose, betas=shape_param)
        
        mesh_neutral_pose = output.vertices[0].detach() * self.hand_rescaling # template hand pose
        mesh_neutral_pose_upsampled = mano.upsample_mesh(mesh_neutral_pose).detach() # template hand pose
        joint_neutral_pose = output.joints[0][:mano.joint_num,:].detach() * self.hand_rescaling # template hand pose

        # compute transformation matrix for making 大 pose to zero pose
        neutral_hand_pose = neutral_hand_pose.view(len(mano.joint_part['hand'])-1,3)
        #zero_hand_pose = zero_hand_pose.view(len(smpl_x.joint_part['lhand']),3)
        neutral_hand_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(neutral_hand_pose)))
        pose = torch.cat((zero_pose, neutral_hand_pose_inv)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_neutral_pose = batch_rigid_transform(pose[None,:,:,:], joint_neutral_pose[None,:,:], self.mano_layer.parents)
        transform_mat_neutral_pose = transform_mat_neutral_pose[0].detach()
        return mesh_neutral_pose_upsampled, mesh_neutral_pose, joint_neutral_pose, transform_mat_neutral_pose

    def get_zero_pose_hand(self, return_mesh=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        zero_hand_pose = torch.zeros((1,(len(mano.joint_part['hand'])-1)*3)).float().cuda()
        shape_param = self.shape_param
        if shape_param.dim()==1:
            shape_param = shape_param[None,:]
        
        #output = self.mano_layer(global_orient=zero_pose, hand_pose=zero_hand_pose, betas=shape_param, transl=self.hand_trans)
        output = self.mano_layer(global_orient=zero_pose, hand_pose=zero_hand_pose, betas=shape_param)
        
        joint_zero_pose = output.joints[0][:mano.joint_num,:].detach() * self.hand_rescaling # zero pose hand
        if not return_mesh:
            return joint_zero_pose
        else: 
            mesh_zero_pose = output.vertices[0].detach() * self.hand_rescaling # zero pose hand
            mesh_zero_pose_upsampled = mano.upsample_mesh(mesh_zero_pose).detach() # zero pose hand
            return mesh_zero_pose_upsampled, mesh_zero_pose, joint_zero_pose

    def get_transform_mat_joint(self, transform_mat_neutral_pose, joint_zero_pose, mano_param):
        # 1. 大 pose -> zero pose. no need for our implementation
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. zero pose -> image pose
        root_pose = mano_param['root_pose'].view(1,3)
        hand_pose = mano_param['hand_pose'].view(len(mano.joint_part['hand'])-1,3)
        trans = mano_param['trans'].view(1,3)

        # forward kinematics
        pose = torch.cat((root_pose, hand_pose))
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.mano_layer.parents)
        #transform_mat_joint_2 = transform_mat_joint_2[0].detach()
        transform_mat_joint_2 = transform_mat_joint_2[0]
        
        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        transform_mat_joint = torch.bmm(transform_mat_joint_2, transform_mat_joint_1)
        return transform_mat_joint

    def get_transform_mat_vertex(self, transform_mat_joint, nn_vertex_idxs):
        skinning_weight = self.skinning_weight[nn_vertex_idxs,:]
        transform_mat_vertex = torch.matmul(skinning_weight, transform_mat_joint.view(mano.joint_num,16)).view(mano.vertex_num_upsampled,4,4)
        return transform_mat_vertex

    def lbs(self, xyz, transform_mat_vertex, trans):
        xyz = torch.cat((xyz, torch.ones_like(xyz[:,:1])),1) # 大 pose. xyz1
        xyz = torch.bmm(transform_mat_vertex, xyz[:,:,None]).view(mano.vertex_num_upsampled,4)[:,:3]
        xyz = xyz + trans
        return xyz

    def capture(self):
        if self.optimizer is None:
            optim_dict = None
        else:
            optim_dict = self.optimizer.state_dict()
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            #self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            optim_dict,
            self.spatial_lr_scale,
            self.shape_param,
            self.mean_offset,
            self.scale_offset,
            self.mean_offset_offset,
            #self.rgb_offset, # this one is obsolete.
            self.f_dc_offset,
            self.f_rest_offset,
            self.hand_trans,
        )
    
    def restore(self, model_args, training_args, mano_param_dict=None):
        if len(model_args)==17:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            #self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self.shape_param,
            self.mean_offset,
            self.scale_offset,
            self.mean_offset_offset,
            self.rgb_offset, # this one is obsolete.
            #self.f_dc_offset,
            #self.f_rest_offset,
            self.hand_trans) = model_args
        else:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            #self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self.shape_param,
            self.mean_offset,
            self.scale_offset,
            self.mean_offset_offset,
            #self.rgb_offset, # this one is obsolete.
            self.f_dc_offset,
            self.f_rest_offset,
            self.hand_trans) = model_args

        opacities = torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device=self._xyz.device)
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        if training_args is not None:
            self.training_setup(training_args, mano_param_dict)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if self.optimizer is not None and opt_dict is not None:
            print('load hand optimizer')
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        scaling = self._scaling + self.scale_offset
        if self.isotropic:
            scaling = scaling.repeat(1,3)
        return self.scaling_activation(scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        mano_param = {'root_pose': matrix_to_axis_angle(rotation_6d_to_matrix(self.root_pose)),
                    'hand_pose': matrix_to_axis_angle(rotation_6d_to_matrix(self.hand_pose)),
                    'trans': self.trans}
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_hand(use_id_info=True)
        joint_zero_pose = self.get_zero_pose_hand()

        mean_offset = self.mean_offset
        mean_3d = mesh_neutral_pose + mean_offset
        
        mean_offset_offset, scale_offset = self.forward_geo()
        mean_combined_offset, mean_offset_offset = self.get_mean_offset_offset(mano_param, mean_offset_offset)
        mean_3d_refined = mean_3d + mean_combined_offset

        # get nearest vertex
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        nn_vertex_idxs = knn_points(mean_3d[None,:,:], mesh_neutral_pose_wo_upsample[None,:,:], K=1, return_nn=True).idx[0,:,0] # dimension: mano.vertex_num_upsampled
        nn_vertex_idxs = self.lr_idx_to_hr_idx(nn_vertex_idxs)
        #mask = (self.is_rhand + self.is_lhand + self.is_face) > 0
        #nn_vertex_idxs[mask] = torch.arange(mano.vertex_num_upsampled).cuda()[mask]
        nn_vertex_idxs = torch.arange(mano.vertex_num_upsampled).cuda()

        # get transformation matrix of the nearest vertex and perform lbs
        transform_mat_joint = self.get_transform_mat_joint(transform_mat_neutral_pose, joint_zero_pose, mano_param)
        transform_mat_vertex = self.get_transform_mat_vertex(transform_mat_joint, nn_vertex_idxs)

        #mean_3d = self.lbs(mean_3d, transform_mat_vertex, mano_param['trans']) # posed with mano_param
        mean_3d = self.lbs(mean_3d, transform_mat_vertex, mano_param['trans']+self.hand_trans) # posed with mano_param
        #mean_3d_refined = self.lbs(mean_3d_refined, transform_mat_vertex, mano_param['trans']) # posed with mano_param
        mean_3d_refined = self.lbs(mean_3d_refined, transform_mat_vertex, mano_param['trans']+self.hand_trans) # posed with mano_param

        #mean_3d_refined_rescaled = self.hand_rescaling * mean_3d_refined
        mean_3d_refined_rescaled = mean_3d_refined
        self._xyz = mean_3d_refined_rescaled.detach()
        """
        # debug. check whether mean_combined_offset is the same as posed mano vertices
        debug_output = self.mano_layer(global_orient=self.root_pose, hand_pose=self.hand_pose.reshape(1,45), betas=self.shape_param, transl=self.hand_trans)
        debug_vertices = debug_output.vertices[0]
        debug_vertices_upsampled = mano.upsample_mesh(debug_vertices)
        print(f"mean_3d_refined.shape: {mean_3d_refined.shape}, debug_vertices_upsampled.shape: {debug_vertices_upsampled.shape}")
        print(f"mano_param['trans']: {mano_param['trans']}, self.hand_trans: {self.hand_trans}")
        hand_v_diff = torch.abs(mean_3d_refined - debug_vertices_upsampled)
        print(f"hand_v_diff.amin(): {hand_v_diff.amin()}, hand_v_diff.mean(): {hand_v_diff.mean()} hand_v_diff.amax(): {hand_v_diff.amax()}")
        print(f"hand_v_diff.mean(dim=0): {hand_v_diff.mean(dim=0)}, hand_v_diff.amax(dim=0): {hand_v_diff.amax(dim=0)}")
        """
        return mean_3d_refined_rescaled

    @property
    def get_features(self):
        features_dc = self._features_dc + self.f_dc_offset
        features_rest = self._features_rest + self.f_rest_offset
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        #return self.opacity_activation(self._opacity)
        return self._opacity
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def forward_geo(self):
        if self.learn_lbs_offset:
            mean_offset_offset = self.mean_offset_offset
        else:
            mean_offset_offset = None
        scale_offset = self.scale_offset
        return mean_offset_offset, scale_offset

    def get_mean_offset_offset(self, mano_param, mean_offset_offset):
        # poses from mano parameters
        hand_pose = mano_param['hand_pose'].view(len(mano.joint_part['hand'])-1,3)#.detach()
        pose = hand_pose

        # mano pose-dependent vertex offset
        pose = (axis_angle_to_matrix(pose) - torch.eye(3)[None,:,:].float().cuda()).view(1,(mano.joint_num-1)*9)
        #mano_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(mano.vertex_num_upsampled,3)
        mano_pose_offset = torch.matmul(pose, self.pose_dirs).view(mano.vertex_num_upsampled,3)

        # combine it with regressed mean_offset_offset
        # for face and hands, use mano offset
        if self.learn_lbs_offset:
            output = mean_offset_offset + mano_pose_offset
        else:
            output = mano_pose_offset
        return output, mean_offset_offset

    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx    

    def assign_mano_params(self, mano_params):
        self.trans = torch.zeros((1,3), device="cuda", dtype=torch.float32)
        if self.learn_mano_param:
            #self.root_pose = mano_params['root_pose']
            self.root_pose = matrix_to_rotation_6d(axis_angle_to_matrix(mano_params['root_pose']))
            self.hand_pose = matrix_to_rotation_6d(axis_angle_to_matrix(mano_params['hand_pose']))
            #self.trans = mano_params['trans']
            self.trans.requires_grad = True
        else:
            #self.root_pose = mano_params['root_pose'].detach()
            self.root_pose = matrix_to_rotation_6d(axis_angle_to_matrix(mano_params['root_pose'].detach()))
            self.hand_pose = matrix_to_rotation_6d(axis_angle_to_matrix(mano_params['hand_pose'].detach()))
            #self.trans = mano_params['trans'].detach()

    def assign_hand_trans(self, mano_params, shape_param=None):
        root_pose = mano_params['root_pose'].detach()
        hand_pose = mano_params['hand_pose'].detach().view(1,-1)
        if shape_param is None:
            shape_param = self.shape_param

        output = self.mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=shape_param)
        vertices = output.vertices[0] * self.hand_rescaling

        # re-center with hand-specific translation
        self.hand_trans = -vertices.mean(dim=0, keepdim=True).detach()

    def create_from_mano(self, mano_params, shape_param, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        self.assign_mano_params(mano_params)
        # get posed mano mesh
        root_pose = mano_params['root_pose'].detach()
        hand_pose = mano_params['hand_pose'].detach().view(1,-1)

        if shape_param is None:
            shape_param = self.shape_param

        self.assign_hand_trans(mano_params, shape_param)

        output = self.mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=shape_param, transl=self.hand_trans)
        #output = self.mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=shape_param)
        vertices = output.vertices[0] * self.hand_rescaling

        vertices_upsampled = mano.upsample_mesh(vertices)

        self.vert_norms = None

        #self.hand_rescaling = 8.0
        fused_point_cloud = vertices_upsampled
        #colors = torch.rand((vertices_upsampled.shape[0], 3), dtype=torch.float32).cuda()*0.0001 + 0.4980392156862745
        colors = torch.zeros((vertices_upsampled.shape[0], 3), dtype=torch.float32).cuda()
        fused_color = RGB2SH(colors)
        if self.hand_color_init == 'default':
            fused_color[:,0] = 0.1 #0.99
            fused_color[:,1] = 0.02 #0.21
            fused_color[:,2] = -0.01 #-0.05
        elif self.hand_color_init == 'east_asian':
            fused_color[:,0] = 1.41101228
            fused_color[:,1] = 0.79934193
            fused_color[:,2] = 0.45180196

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        #print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(vertices_upsampled), 0.0000001)
        if self.isotropic:
            scales = torch.log(torch.sqrt(dist2))[...,None]
        else:
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = self.random_quaternion(fused_point_cloud.shape[0])
        # rots[:, 0] = 1

        # import pdb;pdb.set_trace()
        #opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        opacities = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False)) # this is used for downstream rendering. we do not directly fit this.
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # [49281, 1, 3]
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # [49281, 0, 3]
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        
        self.shape_param = nn.Parameter(shape_param.requires_grad_(True))

        mean_offset = torch.zeros_like(fused_point_cloud)
        self.mean_offset = nn.Parameter(mean_offset.requires_grad_(True))
        scale_offset = torch.zeros_like(scales)
        self.scale_offset = nn.Parameter(scale_offset.requires_grad_(True))

        if self.learn_lbs_offset:
            mean_offset_offset = torch.zeros_like(fused_point_cloud)
            self.mean_offset_offset = nn.Parameter(mean_offset_offset.requires_grad_(True))
        else:
            self.mean_offset_offset = None
        f_dc_offset = torch.zeros_like(self._features_dc)
        self.f_dc_offset = nn.Parameter(f_dc_offset.requires_grad_(True))
        f_rest_offset = torch.zeros_like(self._features_rest)
        self.f_rest_offset = nn.Parameter(f_rest_offset.requires_grad_(True))

    # run before initializing an optimizer. this is used by hoi gaussian model
    def get_param_list_with_setup_prefix(self, training_args, mano_param_dict):
        self.percent_dense = training_args.percent_dense
        xyz_shape = self._xyz.shape
        self.xyz_gradient_accum = torch.zeros((xyz_shape[0], 1), device="cuda")
        self.denom = torch.zeros((xyz_shape[0], 1), device="cuda")

        l = [
            #{'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            #{'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.learn_mano_param:
            l += mano_param_dict.get_optimizable_params(training_args.mano_param_lr)
        l += [{'params': [self.shape_param], 'lr': training_args.mano_param_lr, "name": "mano_shape"}]

        l += [
            {'params': [self.mean_offset], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "mean_offset"},
            {'params': [self.scale_offset], 'lr': training_args.scaling_lr, "name": "scale_offset"},]
        if self.learn_lbs_offset:
            l += [{'params': [self.mean_offset_offset], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "mean_offset_offset"},]
        l += [{'params': [self.f_dc_offset], 'lr':training_args.feature_lr, "name": "f_dc_offset"},
            {'params': [self.f_rest_offset], 'lr':training_args.feature_lr / 20.0, "name": "f_rest_offset"},
        ]
        return l

    def training_setup(self, training_args, mano_param_dict):

        l = self.get_param_list_with_setup_prefix(training_args, mano_param_dict)

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr,
                                                    lr_final=training_args.rotation_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

        self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr,
                                                    lr_final=training_args.scaling_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

        self.feature_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr,
                                                    lr_final=training_args.feature_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["mean_offset", "mean_offset_offset"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_feature_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["f_dc", "f_dc_offset"]:
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_rotation_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_scaling_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["scaling", "scale_offset"]:
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        #l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        #l.append('shape')
        l += ['x_offset', 'y_offset', 'z_offset']
        for i in range(self.scale_offset.shape[1]):
            l.append('scaleoffset_{}'.format(i))
        if self.learn_lbs_offset:
            l += ['x_offset_offset', 'y_offset_offset', 'z_offset_offset']
        for i in range(self.f_dc_offset.shape[1]*self.f_dc_offset.shape[2]):
            l.append('f_dc_offset_{}'.format(i))
        for i in range(self.f_rest_offset.shape[1]*self.f_rest_offset.shape[2]):
            l.append('f_restoffset_{}'.format(i))
        return l

    # save ply with GaussianDreamerPro format
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        shape_param = self.shape_param.detach().cpu().numpy()

        mean_offset = self.mean_offset.detach().cpu().numpy()
        scale_offset = self.scale_offset.detach().cpu().numpy()
        if self.learn_lbs_offset:
            mean_offset_offset = self.mean_offset_offset.detach().cpu().numpy()
        f_dc_offset = self.f_dc_offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_offset = self.f_rest_offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        #attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        
        if self.learn_lbs_offset:
            attr_items_cat = (xyz, normals, f_dc, f_rest, scale, rotation, 
                            mean_offset, scale_offset, mean_offset_offset, f_dc_offset, f_rest_offset)
        else:
            attr_items_cat = (xyz, normals, f_dc, f_rest, scale, rotation, 
                            mean_offset, scale_offset, f_dc_offset, f_rest_offset)
        attributes = np.concatenate(attr_items_cat, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        shape_param = shape_param.T
        shape_elem = np.empty(shape_param.shape[0], dtype=[('betas', 'f4')])
        shape_elem[:] = list(map(tuple, shape_param))
        shape_el = PlyElement.describe(shape_elem, 'betas')

        hand_trans = self.hand_trans.detach().cpu().numpy().T
        #print(f'hand_trans.shape at save_ply: {hand_trans.shape}')$ [3,1]
        trans_elem = np.empty(hand_trans.shape[0], dtype=[('hand_trans', 'f4')])
        trans_elem[:] = list(map(tuple, hand_trans))
        trans_el = PlyElement.describe(trans_elem, 'hand_trans')
        PlyData([el, shape_el, trans_el]).write(path)

    # load ply with GaussianDreamerPro format
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        #opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        mean_offset = np.stack((np.asarray(plydata.elements[0]["x_offset"]),
                                np.asarray(plydata.elements[0]["y_offset"]),
                                np.asarray(plydata.elements[0]["z_offset"])),  axis=1)
        
        scale_offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scaleoffset_")]
        scale_offset_names = sorted(scale_offset_names, key = lambda x: int(x.split('_')[-1]))
        scale_offset = np.zeros((xyz.shape[0], len(scale_offset_names)))
        for idx, attr_name in enumerate(scale_offset_names):
            scale_offset[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if self.learn_lbs_offset:
            mean_offset_offset = np.stack((np.asarray(plydata.elements[0]["x_offset_offset"]),
                                        np.asarray(plydata.elements[0]["y_offset_offset"]),
                                        np.asarray(plydata.elements[0]["z_offset_offset"])),  axis=1)
        
        f_dc_offset = np.zeros((xyz.shape[0], 3, 1))
        f_dc_offset[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_offset_0"])
        f_dc_offset[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_offset_1"])
        f_dc_offset[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_offset_2"])

        f_rest_offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_restoffset_")]
        f_rest_offset_names = sorted(f_rest_offset_names, key = lambda x: int(x.split('_')[-1]))
        assert len(f_rest_offset_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        f_rest_offset = np.zeros((xyz.shape[0], len(f_rest_offset_names)))
        for idx, attr_name in enumerate(f_rest_offset_names):
            f_rest_offset[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        f_rest_offset = f_rest_offset.reshape((f_rest_offset.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        self.mean_offset = nn.Parameter(torch.tensor(mean_offset, dtype=torch.float, device="cuda").requires_grad_(True))
        self.scale_offset = nn.Parameter(torch.tensor(scale_offset, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.learn_lbs_offset:
            self.mean_offset_offset = nn.Parameter(torch.tensor(mean_offset_offset, dtype=torch.float, device="cuda").requires_grad_(True))
        self.f_dc_offset = nn.Parameter(torch.tensor(f_dc_offset, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.f_rest_offset = nn.Parameter(torch.tensor(f_rest_offset, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        shape_param = np.asarray(plydata.elements[1]["betas"])[None,:]
        self.shape_param = nn.Parameter(torch.tensor(shape_param, dtype=torch.float, device='cuda').requires_grad_(True))

        # assign hand trans
        hand_trans = np.asarray(plydata.elements[2]["hand_trans"])[None,:]
        #print(f'hand_trans.shape at load_ply: {hand_trans.shape}')
        self.hand_trans = torch.tensor(hand_trans, dtype=torch.float, device='cuda', requires_grad=False)


    def get_lap_loss(self, loss_fn):
        # lap_mean
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_hand_pose = mano.neutral_hand_pose.view(1,-1).cuda() # tempalte pose
        shape_param = self.shape_param
        if shape_param.dim()==1:
            shape_param = shape_param[None,:]
        output = self.mano_layer(global_orient=zero_pose, hand_pose=neutral_hand_pose, betas=shape_param, transl=self.hand_trans)
        mesh_neutral_pose = output.vertices[0]
        mesh_neutral_pose = mano.upsample_mesh(mesh_neutral_pose)
        mesh_neutral_pose = mesh_neutral_pose[None,:,:].detach() * self.hand_rescaling

        mean_offset = self.mean_offset
        mean_learned = (mesh_neutral_pose + mean_offset)
        mean_loss = loss_fn(mean_learned, mesh_neutral_pose).mean() * 100000
        if self.learn_lbs_offset:
            mean_offset_offset = self.mean_offset_offset
            mean_offset_learned = (mesh_neutral_pose + mean_offset + mean_offset_offset)
            mean_loss = mean_loss + (loss_fn(mean_offset_learned, mesh_neutral_pose).mean()) * 100000
        
        # lap_scale
        scale = self.scaling_activation(self._scaling).unsqueeze(0)
        scale_refined = self.get_scaling.unsqueeze(0)
        scale_loss = loss_fn(scale, None).mean() + loss_fn(scale_refined, None).mean() * 100000

        # lap_rgb
        shs = torch.cat((self._features_dc, self._features_rest), dim=1)
        rgb = SH2RGB(shs[:,0].float()).unsqueeze(0)
        shs_refined = self.get_features
        rgb_refined = SH2RGB(shs_refined[:,0].float()).unsqueeze(0)
        rgb_loss = loss_fn(rgb, None).mean() * 10 + loss_fn(rgb_refined, None).mean() * 100000

        loss = mean_loss + scale_loss + rgb_loss
        
        return loss
