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
from tqdm import tqdm
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes

from torch.nn import functional as F

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.mano import mano
from utils.smplx.smplx.lbs import batch_rigid_transform
from .hand_gaussian_model import HandGaussianModel
from .layer import make_linear_layers, Sigmoid_0origin


class HandGaussianTriplaneWrapper(HandGaussianModel):

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

        # tirplane
        device = 'cuda:0'
        #self.triplane = nn.Parameter(torch.zeros((3,*cfg.triplane_shape), dtype=torch.float32, device=device))
        #self.triplane_face = nn.Parameter(torch.zeros((3,*cfg.triplane_shape), dtype=torch.float32, device=device))
        self.triplane = nn.Parameter((torch.rand((3,*cfg.triplane_shape), dtype=torch.float32, device=device, requires_grad=True)-0.5)*1.0e-6)
        #self.triplane_face = nn.Parameter((torch.rand((3,*cfg.triplane_shape), dtype=torch.float32, device=device, requires_grad=True)-0.5)*1.0e-6)

        self.geo_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128], use_gn=True).to(device)
        self.mean_offset_net = make_linear_layers([128, 3], relu_final=False).to(device)
        self.scale_net = make_linear_layers([128, 1], relu_final=False).to(device)
        #self.scale_net = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid()).to(device) # GaussianAvatar style
        self.geo_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(len(mano.joint_part['hand'])-1)*6, 128, 128, 128], use_gn=True).to(device)

        self.learn_lbs_offset = cfg.learn_lbs_offset
        if self.learn_lbs_offset:
            self.mean_offset_offset_net = make_linear_layers([128, 3], relu_final=False).to(device)
        else:
            self.mean_offset_offset_net = nn.Identity()
        self.scale_offset_net = make_linear_layers([128, 1], relu_final=False).to(device)
        #self.scale_offset_net = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid()).to(device) # GaussianAvatar style. this only produce nonnegative values
        #self.scale_offset_net = nn.Sequential(nn.Linear(128, 1), Sigmoid_0origin()).to(device) # GaussianAvatar style
        
        #self.rot_net = make_linear_layers([128, 1], relu_final=False).to(device)
        #self.rot_offset_net = make_linear_layers([128, 1], relu_final=False).to(device)
        
        f_rest_dim = (self.max_sh_degree + 1) ** 2 - 1
        print(f'f_rest_dim: {f_rest_dim}')
        self.f_dc_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128, 3], relu_final=False, use_gn=True).to(device)
        self.f_dc_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(len(mano.joint_part['hand'])-1)*6+3, 128, 128, 128, 3], relu_final=False, use_gn=True).to(device)
        if f_rest_dim > 0:
            self.f_rest_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128, f_rest_dim * 3], relu_final=False, use_gn=True).to(device)
            self.f_rest_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(len(mano.joint_part['hand'])-1)*6+3, 128, 128, 128, f_rest_dim * 3], relu_final=False, use_gn=True).to(device)
        else:
            self.f_rest_net = nn.Identity()
            self.f_rest_offset_net = nn.Identity()

        # initialize network weights
        self.init_net_weights()

        self.triplane_shape_3d = cfg.triplane_shape_3d
        """
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        """
        self.optimizer = None
        #self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.isotropic = cfg.isotropic
        self.learn_mano_param = cfg.learn_mano_param

        self.mano_layer = copy.deepcopy(mano.layer[cfg.mano_rhand]).cuda()
        self.shape_param = nn.Parameter(mano.shape_param.float().cuda())
        #self.shape_param = nn.Parameter(torch.zeros((1,10), dtype=torch.float32, device='cuda'))

        # exavatar style, but without triplane
        #self.mean_offset = torch.empty(0) # offset for template pose
        #self.scale_offset = torch.empty(0) # offset for articulated pose

        #self.mean_offset_offset = torch.empty(0) # offset for articulated pose. this is not used since we directly use mano lbs weight

        #self.rgb_offset = torch.empty(0) # offset for articulated pose

        # upsample mesh and other assets
        self.hand_trans = None
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
        self.hand_pose = torch.empty(0)
        self.trans = torch.empty(0)
        self.n_gs = mano.vertex_num_upsampled

    def init_linear_weight(self, net, minv, maxv):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, minv, maxv)
                #nn.init.constant_(m.weight, 0)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def init_net_weights(self):
        weight_scale = 1.0e-6
        networks = [self.geo_net, self.mean_offset_net, self.geo_offset_net]
        if self.learn_lbs_offset:
            networks.append(self.mean_offset_offset_net)
        for network in networks:
            self.init_linear_weight(network, -weight_scale, weight_scale)
        
        # f net
        f_nets = [self.f_dc_net, self.f_rest_net, self.f_dc_offset_net, self.f_rest_offset_net]
        for network in f_nets:
            self.init_linear_weight(network, -weight_scale, weight_scale)

        # scale net
        scale_nets = [self.scale_net, self.scale_offset_net]
        for network in scale_nets:
            self.init_linear_weight(network, -weight_scale, weight_scale)


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
        
        mesh_neutral_pose = output.vertices[0].detach() # template hand pose
        mesh_neutral_pose_upsampled = mano.upsample_mesh(mesh_neutral_pose).detach() # template hand pose
        joint_neutral_pose = output.joints[0][:mano.joint_num,:].detach() # template hand pose

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
        
        joint_zero_pose = output.joints[0][:mano.joint_num,:].detach() # zero pose hand
        if not return_mesh:
            return joint_zero_pose
        else: 
            mesh_zero_pose = output.vertices[0].detach() # zero pose hand
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
        transform_mat_joint_2 = transform_mat_joint_2[0].detach()
        
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
        return (
            self.active_sh_degree,
            #self._xyz,
            #self._features_dc,
            #self._features_rest,
            #self._scaling,
            #self._rotation,
            #self._opacity,
            #self.max_radii2D,
            #self.xyz_gradient_accum,
            #self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.shape_param,
            #self.mean_offset,
            #self.scale_offset,
            #self.mean_offset_offset,
            #self.rgb_offset,
            self.triplane,
            #self.triplane_face,
            self.geo_net.state_dict(),
            self.mean_offset_net.state_dict(),
            self.scale_net.state_dict(),
            self.geo_offset_net.state_dict(),
            self.mean_offset_offset_net.state_dict(),
            self.scale_offset_net.state_dict(),
            #self.rot_net.state_dict(),
            #self.rot_offset_net.state_dict(),
            self.f_dc_net.state_dict(),
            self.f_rest_net.state_dict(),
            self.f_dc_offset_net.state_dict(),
            self.f_rest_offset_net.state_dict(),
            self.root_pose,
            self.hand_pose,
            self.trans
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        #self._xyz, 
        #self._features_dc, 
        #self._features_rest,
        #self._scaling, 
        #self._rotation, 
        #self._opacity,
        #self.max_radii2D, 
        #xyz_gradient_accum, 
        #denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.shape_param,
        #self.mean_offset,
        #self.scale_offset,
        #self.mean_offset_offset,
        #self.rgb_offset,
        self.triplane,
        #self.triplane_face,
        geo_net,
        mean_offset_net,
        scale_net,
        geo_offset_net,
        mean_offset_offset_net,
        scale_offset_net,
        #rot_net,
        #rot_offset_net,
        f_dc_net,
        f_rest_net,
        f_dc_offset_net,
        f_rest_offset_net,
        root_pose,
        hand_pose,
        trans) = model_args

        self.geo_net.load_state_dict(geo_net)
        self.mean_offset_net.load_state_dict(mean_offset_net)
        self.scale_net.load_state_dict(scale_net)
        self.geo_offset_net.load_state_dict(geo_offset_net)
        if self.learn_lbs_offset:
            self.mean_offset_offset_net.load_state_dict(mean_offset_offset_net)
        self.scale_offset_net.load_state_dict(scale_offset_net)
        #self.rot_net.load_state_dict(rot_net)
        #self.rot_offset_net.load_state_dict(rot_offset_net)
        self.f_dc_net.load_state_dict(f_dc_net)
        self.f_rest_net.load_state_dict(f_rest_net)
        self.f_dc_offset_net.load_state_dict(f_dc_offset_net)
        self.f_rest_offset_net.load_state_dict(f_rest_offset_net)
        
        learn_mano_param = self.learn_mano_param
        self.root_pose = nn.Parameter(root_pose.requires_grad_(learn_mano_param))
        self.hand_pose = nn.Parameter(hand_pose.requires_grad_(learn_mano_param))
        self.trans = nn.Parameter(trans.requires_grad_(learn_mano_param))

        self.n_gs = mano.vertex_num_upsampled
        device = self.triplane.device
        
        #rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(self.n_gs,1,1)) # constant rotation
        #rotation = F.normalize(rotation)
        rotation = torch.zeros((self.n_gs, 4), dtype=torch.float32, device=device)
        rotation[:,0] = 1.
        self._rotation = nn.Parameter(rotation.requires_grad_(False))
        
        opacities = torch.ones((self.n_gs, 1), dtype=torch.float32, device=device)
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def extract_tri_feature(self):
        ## 1. triplane features of all vertices
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_mesh
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / (self.triplane_shape_3d[0]/2)
        y = xyz[:,1] / (self.triplane_shape_3d[1]/2)
        z = xyz[:,2] / (self.triplane_shape_3d[2]/2)
        
        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # smpl_x.vertex_num_upsampled, cfg.triplane_shape[0]*3

        return tri_feat

    def forward_geo_network(self, tri_feat, mano_param):
        # pose from mano parameters (only use body pose as face/hand poses are not diverse in the training set)
        hand_pose = mano_param['hand_pose'].view(len(mano.joint_part['hand'])-1,3)

        # combine pose with triplane feature
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(hand_pose)).view(1,(len(mano.joint_part['hand'])-1)*6).repeat(mano.vertex_num_upsampled,1) # without root pose
        feat = torch.cat((tri_feat, pose.detach()),1)

        # forward to geometry networks
        geo_offset_feat = self.geo_offset_net(feat)
        if self.learn_lbs_offset:
            mean_offset_offset = self.mean_offset_offset_net(geo_offset_feat) # pose-dependent mean offset of Gaussians
        else:
            mean_offset_offset = None
        scale_offset = self.scale_offset_net(geo_offset_feat) # pose-dependent scale of Gaussians
        return mean_offset_offset, scale_offset


    def get_mean_offset_offset(self, mano_param, mean_offset_offset):
        # poses from mano parameters
        hand_pose = mano_param['hand_pose'].view(len(mano.joint_part['hand'])-1,3)#.detach()
        pose = hand_pose

        # mano pose-dependent vertex offset
        pose = (axis_angle_to_matrix(pose) - torch.eye(3)[None,:,:].float().cuda()).view(1,(mano.joint_num-1)*9)
        mano_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(mano.vertex_num_upsampled,3)

        # combine it with regressed mean_offset_offset
        # for face and hands, use mano offset
        if self.learn_lbs_offset:
            output = mean_offset_offset + mano_pose_offset
        else:
            output = mano_pose_offset
        return output, mean_offset_offset

    def f_sh_net(self, tri_feat):
        f_dc = self.f_dc_net(tri_feat).unsqueeze(1)
        #if self.max_sh_degree > 0:
        if not isinstance(self.f_rest_net, torch.nn.Identity):
            n_gs = f_dc.shape[0]
            f_rest = self.f_rest_net(tri_feat).reshape(n_gs, -1, 3)
            f_sh = torch.cat((f_dc, f_rest), dim=1)
        else:
            f_sh = f_dc
        return f_sh

    def f_sh_offset_net(self, tri_feat, mano_param, xyz):
        # pose from mano parameters (only use body pose as face/hand poses are not diverse in the training set)
        hand_pose = mano_param['hand_pose'].view(len(mano.joint_part['hand'])-1,3)
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(hand_pose)).view(1,(len(mano.joint_part['hand'])-1)*6).repeat(mano.vertex_num_upsampled,1) # without root pose
       
        # per-vertex normal in world coordinate system
        with torch.no_grad():
            normal = Meshes(verts=xyz[None], faces=torch.LongTensor(mano.face_upsampled).cuda()[None]).verts_normals_packed().reshape(mano.vertex_num_upsampled,3)
            #is_cavity = self.is_cavity[:,None].float()
            #normal = normal * (1 - is_cavity) + (-normal) * is_cavity # cavity has opposite normal direction in the template mesh

        # forward to rgb network
        feat = torch.cat((tri_feat, pose.detach(), normal.detach()),1)
        f_dc_offset = self.f_dc_offset_net(feat).unsqueeze(1) # pose-and-view-dependent rgb offset of Gaussians
        #if self.max_sh_degree > 0:
        if not isinstance(self.f_rest_offset_net, torch.nn.Identity):
            n_gs = f_dc_offset.shape[0]
            f_rest_offset = self.f_rest_offset_net(feat).reshape(n_gs, -1, 3)
            f_sh_offset = torch.cat((f_dc_offset, f_rest_offset), dim=1)
        else:
            f_sh_offset = f_dc_offset
        return f_sh_offset

    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx   

    def set_train(self):
        self.geo_net.train()
        self.mean_offset_net.train()
        self.scale_net.train()
        self.geo_offset_net.train()
        if self.learn_lbs_offset:
            self.mean_offset_offset_net.train()
        self.f_dc_net.train()
        self.f_dc_offset_net.train()
        if not isinstance(self.f_rest_net, nn.Identity):
            self.f_rest_net.train()
            self.f_rest_offset_net.train()
    
    def set_eval(self):
        self.geo_net.eval()
        self.mean_offset_net.eval()
        self.scale_net.eval()
        self.geo_offset_net.eval()
        if self.learn_lbs_offset:
            self.mean_offset_offset_net.eval()
        self.f_dc_net.eval()
        self.f_dc_offset_net.eval()
        if not isinstance(self.f_rest_net, nn.Identity):
            self.f_rest_net.eval()
            self.f_rest_offset_net.eval()

    # tb_writer and iter are used only for debugging
    def forward(self, scaling_modifier=1, tb_writer=None, iter=-1):
        mano_param = {'root_pose': self.root_pose,
                    'hand_pose': self.hand_pose,
                    'trans': self.trans}
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_hand(use_id_info=True)
        joint_zero_pose = self.get_zero_pose_hand()

        # extract triplane feature
        tri_feat = self.extract_tri_feature()
      
        # get Gaussian assets
        geo_feat = self.geo_net(tri_feat)
        mean_offset = self.mean_offset_net(geo_feat) # mean offset of Gaussians
        scale_raw = self.scale_net(geo_feat) # scale of Gaussians

        f_sh = self.f_sh_net(tri_feat)
        #f_sh_out = F.tanh(f_sh) * 1.772453850905516 # valid only if max_sh_degree==0
        #f_sh_out = torch.clamp(f_sh, -1.772453850905516, 1.772453850905516)
        f_sh_out = (3.545 * F.sigmoid(f_sh) - 1.7725) # BrightDreamer style
        #rgb = self.rgb_net(tri_feat) # rgb of Gaussians
        mean_3d = mesh_neutral_pose + mean_offset # 大 pose
 
        # get pose-dependent Gaussian assets
        mean_offset_offset, scale_offset = self.forward_geo_network(tri_feat, mano_param)
        #scale, scale_refined = torch.exp(scale).repeat(1,3), torch.exp(scale+scale_offset).repeat(1,3) # exavatar style
        #scale, scale_refined = scale_raw.repeat(1,3), (scale_raw+scale_offset).repeat(1,3)
        #scale, scale_refined = F.sigmoid(scale_raw).repeat(1,3), F.sigmoid(scale_raw+scale_offset).repeat(1,3)
        #scale = torch.clamp(scale_raw, 1.0e-4).repeat(1,3)
        #scale_refined = torch.clamp(scale_raw+scale_offset, 1.0e-4).repeat(1,3)
        scale = torch.exp(6*F.sigmoid(scale_raw)-9).repeat(1,3) # BrightDreamer style
        scale_refined = torch.exp(6*F.sigmoid(scale_raw+scale_offset)-9).repeat(1,3) # BrightDreamer style
        
        mean_combined_offset, mean_offset_offset = self.get_mean_offset_offset(mano_param, mean_offset_offset)
        mean_3d_refined = mean_3d + mean_combined_offset # 大 pose
        
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

        # camera coordinate system -> world coordinate system
        """
        if not is_world_coord:
            mean_3d = torch.matmul(torch.inverse(cam_param['R']), (mean_3d - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
            mean_3d_refined = torch.matmul(torch.inverse(cam_param['R']), (mean_3d_refined - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        """

        # forward to rgb network
        f_sh_offset = self.f_sh_offset_net(tri_feat, mano_param, mean_3d_refined)
        #f_sh_refined = F.tanh(f_sh + f_sh_offset) * 1.772453850905516 # valid only if max_sh_degree==0
        #f_sh_refined = F.normalize(f_sh + f_sh_offset, p=1, dim=2) * 1.772453850905516
        #f_sh_refined = torch.clamp(f_sh + f_sh_offset, -1.772453850905516, 1.772453850905516)
        #f_sh_refined = f_sh + f_sh_offset
        f_sh_refined = (3.545 * F.sigmoid(f_sh + f_sh_offset) - 1.7725) # BrightDreamer style

        # Gaussians and offsets
        rotation = self._rotation
        #gs_rotmat = quaternion_to_matrix(self._rotation)
        #deformed_gs_rotmat = torch.bmm(transform_mat_vertex[:,:3,:3], gs_rotmat)
        #rotation_refined = matrix_to_quaternion(deformed_gs_rotmat)
        rotation_refined = self._rotation

        opacity = self._opacity
        cov = self.covariance_activation(scale, scaling_modifier, rotation)
        cov_refined = self.covariance_activation(scale_refined, scaling_modifier, rotation)

        if tb_writer is not None:
            tb_writer.add_scalar('triplane/min', self.triplane.amin().item(), iter)
            tb_writer.add_scalar('triplane/max', self.triplane.amax().item(), iter)

            tb_writer.add_scalar('mean_offset/min', mean_offset.amin().item(), iter)
            tb_writer.add_scalar('mean_offset/max', mean_offset.amax().item(), iter)
            tb_writer.add_scalar('mean_combined_offset/min', mean_combined_offset.amin().item(), iter)
            tb_writer.add_scalar('mean_combined_offset/max', mean_combined_offset.amax().item(), iter)

            tb_writer.add_scalar('scale_raw/min', scale_raw.amin().item(), iter)
            tb_writer.add_scalar('scale_raw/max', scale_raw.amax().item(), iter)
            tb_writer.add_scalar('scale/min', scale.amin().item(), iter)
            tb_writer.add_scalar('scale/max', scale.amax().item(), iter)
            tb_writer.add_scalar('scale_offset/min', scale_offset.amin().item(), iter)
            tb_writer.add_scalar('scale_offset/max', scale_offset.amax().item(), iter)
            tb_writer.add_scalar('scale_refined/min', scale_refined.amin().item(), iter)
            tb_writer.add_scalar('scale_refined/max', scale_refined.amax().item(), iter)

            tb_writer.add_scalar('f_sh/min', f_sh.amin().item(), iter)
            tb_writer.add_scalar('f_sh/max', f_sh.amax().item(), iter)
            tb_writer.add_scalar('f_sh_out/min', f_sh_out.amin().item(), iter)
            tb_writer.add_scalar('f_sh_out/max', f_sh_out.amax().item(), iter)
            tb_writer.add_scalar('f_sh_offset/min', f_sh_offset.amin().item(), iter)
            tb_writer.add_scalar('f_sh_offset/max', f_sh_offset.amax().item(), iter)
            tb_writer.add_scalar('f_sh_refined/min', f_sh_refined.amin().item(), iter)
            tb_writer.add_scalar('f_sh_refined/max', f_sh_refined.amax().item(), iter)

        assets = {
                'mean_3d': mean_3d,
                'opacity': opacity,
                'scale': scale,
                'rotation': rotation,
                'f_sh': f_sh_out,
                'cov': cov,
                'active_sh_degree': self.active_sh_degree,
                'max_sh_degree': self.max_sh_degree
                }
        assets_refined = {
                'mean_3d': mean_3d_refined, 
                'opacity': opacity, 
                'scale': scale_refined, 
                'rotation': rotation_refined, 
                'f_sh': f_sh_refined,
                'cov': cov_refined,
                'active_sh_degree': self.active_sh_degree,
                'max_sh_degree': self.max_sh_degree
                }
        offsets = {
                'mean_offset': mean_offset,
                'mean_offset_offset': mean_offset_offset, # None if learn_lbs_offset is False
                'scale_offset': scale_offset,
                'f_sh_offset': f_sh_offset
                }
        return assets, assets_refined, offsets, mesh_neutral_pose

    """
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
        mano_param = {'root_pose': self.root_pose,
                    'hand_pose': self.hand_pose,
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

        self._xyz = mean_3d_refined.detach()

        return mean_3d_refined

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
    """
    def create_from_mano(self, mano_params, shape_param, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

        if self.learn_mano_param:
            self.root_pose = mano_params['root_pose']
            self.hand_pose = mano_params['hand_pose']
            self.trans = mano_params['trans']
        else:
            self.root_pose = mano_params['root_pose'].detach()
            self.hand_pose = mano_params['hand_pose'].detach()
            self.trans = mano_params['trans'].detach()

        # get posed mano mesh
        #root_pose = torch.zeros((1,3), dtype=torch.float32).cuda()
        #hand_pose = mano.neutral_hand_pose.view(1,-1).cuda()
        root_pose = mano_params['root_pose'].detach()
        hand_pose = mano_params['hand_pose'].detach().view(1,-1)

        print(f'at create_from_mano, root_pose.shape: {root_pose.shape}, hand_pose.shape: {hand_pose.shape}')

        if shape_param is None:
            shape_param = self.shape_param
        output = self.mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=shape_param)
        vertices = output.vertices[0]

        # re-center with hand-specific translation
        self.hand_trans = -vertices.mean(dim=0, keepdim=True).requires_grad_(False)
        output = self.mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=shape_param, transl=self.hand_trans)
        #output = self.mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=shape_param)
        vertices = output.vertices[0]
        vertices_upsampled = mano.upsample_mesh(vertices)

        n_gs = mano.vertex_num_upsampled
        #rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(n_gs,1,1)) # constant rotation
        #rotation = F.normalize(rotation)
        rotation = torch.zeros((n_gs, 4), dtype=torch.float32).cuda()
        rotation[:,0] = 1.
        self._rotation = nn.Parameter(rotation.requires_grad_(False))

        opacities = torch.ones((n_gs, 1), dtype=torch.float32).cuda()
        self._opacity = nn.Parameter(opacities.requires_grad_(False))

        self.shape_param = nn.Parameter(shape_param.requires_grad_(True))

    # optimize triplane-related params to output initial values
    def optim_for_init(self, init_pth_path=''):
        if init_pth_path == '':
            init_pth_path = os.path.join('load', 'hand_gs_model_init.pth')
        if os.path.exists(init_pth_path):
            print(f'loading initial hand triplane weights from {init_pth_path}')
            self.load_pth(init_pth_path)
        else:
            lr = 1e-3
            triplane_lr = 1e-5
            position_lr = 1e-5
            feat_lr = 5e-4
            scale_lr = 5e-4
            init_opt_iters = 7000

            print(f'initial random triplane min: {self.triplane.amin()}, max: {self.triplane.amax()}')
            params = [
                {'params': [self.triplane], 'lr': triplane_lr, 'name': 'triplane'},
                #{'params': [self.triplane_face], 'lr': triplane_lr, 'name': 'triplane_face'},
                {'params': self.geo_net.parameters(), 'lr': position_lr, 'name': 'geo_net'},
                {'params': self.mean_offset_net.parameters(), 'lr': position_lr, 'name': 'mean_offset_net'},
                {'params': self.scale_net.parameters(), 'lr': scale_lr, 'name': 'scale_net'},
                {'params': self.geo_offset_net.parameters(), 'lr': position_lr, 'name': 'geo_offset_net'},
                #{'params': self.mean_offset_offset_net.parameters(), 'lr': training_args.position_lr_init * self.spatial_lr_scale},
                {'params': self.scale_offset_net.parameters(), 'lr': scale_lr, 'name': 'scale_offset_net'},
                {'params': self.f_dc_net.parameters(), 'lr': feat_lr, 'name': 'f_dc_net'},
                {'params': self.f_dc_offset_net.parameters(), 'lr': feat_lr, 'name': 'f_dc_offset_net'},
            ]
            if not isinstance(self.f_rest_net, nn.Identity):
                params + [
                    {'params': self.f_rest_net.parameters(), 'lr': feat_lr / 20.0, 'name': 'f_rest_net'},
                    {'params': self.f_rest_offset_net.parameters(), 'lr': feat_lr / 20.0, 'name': 'f_rest_offset_net'},
                ]

            if self.learn_lbs_offset:
                params += [{'params': self.mean_offset_offset_net.parameters(), 'lr': position_lr, 'name': 'mean_offset_offset_net'}]

            #optim = torch.optim.Adam(params, lr=0.0, eps=1e-15, weight_decay=1.0e-6)
            optim = torch.optim.Adam(params, lr=0.0, eps=1e-15)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1000, verbose=True, factor=0.5)
            #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[1000,2000,3000,4000,5000,6000], gamma=0.5)
            loss_fn = torch.nn.MSELoss()

            assets, assets_refined, offsets, mesh_neutral_pose = self.forward()

            mean_offset_gt = torch.zeros_like(offsets['mean_offset']).requires_grad_(False)
            if self.learn_lbs_offset:
                mean_offset_offset_gt = torch.zeros_like(offsets['mean_offset_offset']).requires_grad_(False)
            dist2 = torch.clamp_min(distCUDA2(mesh_neutral_pose), 0.0000001)
            #scale_gt = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # use log if we apply exp on scale result
            scale_gt = torch.sqrt(dist2)[...,None].repeat(1, 3) # exclude log since we do not apply exp during forward pass
            print(f'scale_gt.amin(): {scale_gt.amin()}, scale_gt.amax(): {scale_gt.amax()}')
            scale_offset_gt = torch.zeros_like(assets['scale']).requires_grad_(False)

            #colors = torch.rand((mesh_neutral_pose.shape[0], 3), dtype=torch.float32).cuda()*1.0e-6 + 0.4980392156862745
            colors = torch.zeros((mesh_neutral_pose.shape[0], 3), dtype=torch.float32).cuda() + 0.4980392156862745
            fused_color = RGB2SH(colors)
            f_sh_gt = torch.zeros_like(assets['f_sh']).requires_grad_(False)
            f_sh_gt[:, 0,:3] = fused_color
            print(f'f_sh_gt.shape: {f_sh_gt.shape}')
            print(f'f_sh_gt.amin(): {f_sh_gt.amin()}, f_sh_gt.amax(): {f_sh_gt.amax()}')

            print('optimizing hand gs model for initialization')
            pbar = tqdm(range(init_opt_iters))
            for it in pbar:
                assets, assets_refined, offsets, mesh_neutral_pose = self.forward()

                mean_offset_loss = loss_fn(offsets['mean_offset'], mean_offset_gt)
                if self.learn_lbs_offset:
                    mean_offset_offset_loss = loss_fn(offsets['mean_offset_offset'], mean_offset_offset_gt)
                scale_loss = loss_fn(assets['scale'], scale_gt)
                #scale_offset_loss = loss_fn(offsets['scale_offset'], scale_offset_gt)
                scale_offset_loss = loss_fn(assets_refined['scale'], scale_gt)
                f_sh_loss = loss_fn(assets['f_sh'], f_sh_gt)
                #f_sh_offset_loss = loss_fn(offsets['f_sh_offset'], f_sh_gt)
                f_sh_offset_loss = loss_fn(assets_refined['f_sh'], f_sh_gt)
                loss = mean_offset_loss + scale_loss + scale_offset_loss + f_sh_loss + f_sh_offset_loss
                if self.learn_lbs_offset:
                    loss = loss + mean_offset_offset_loss
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                lr_scheduler.step(loss.item())
                if it % 100 == 0:
                    pbar.set_description(f'[{it}/{init_opt_iters}] total loss: {loss:.4e}, mean_offset_loss: {mean_offset_loss:.3e}, scale_loss: {scale_loss:.3e}, scale_offset_loss: {scale_offset_loss:.3e}, f_sh_loss: {f_sh_loss:.3e}, f_sh_offset_loss: {f_sh_offset_loss:.3e}')
            # save
            self.save_pth(init_pth_path)
            print(f'initially optimized triplane min: {self.triplane.amin()}, max: {self.triplane.amax()}')
            print(f'initial hand gs model is saved to {init_pth_path}')

    # run before initializing an optimizer. this is used by hoi gaussian model
    def get_param_list_with_setup_prefix(self, training_args, mano_param_dict):
        l = [
            {'params': [self.triplane], 'lr': training_args.triplane_lr, 'name': 'triplane', 'weight_decay': training_args.weight_decay},
            #{'params': [self.triplane_face], 'lr': training_args.triplane_lr, 'name': 'triplane_face'},
            {'params': self.geo_net.parameters(), 'lr': training_args.position_lr_init * self.spatial_lr_scale, 'name': 'geo_net', 'weight_decay': training_args.weight_decay},
            {'params': self.mean_offset_net.parameters(), 'lr': training_args.position_lr_init * self.spatial_lr_scale, 'name': 'mean_offset_net', 'weight_decay': training_args.weight_decay},
            {'params': self.scale_net.parameters(), 'lr': training_args.scaling_lr, 'name': 'scale_net', 'weight_decay': training_args.weight_decay},
            {'params': self.geo_offset_net.parameters(), 'lr': training_args.position_lr_init * self.spatial_lr_scale, 'name': 'geo_offset_net', 'weight_decay': training_args.weight_decay},
            #{'params': self.mean_offset_offset_net.parameters(), 'lr': training_args.position_lr_init * self.spatial_lr_scale},
            {'params': self.scale_offset_net.parameters(), 'lr': training_args.scaling_lr, 'name': 'scale_offset_net', 'weight_decay': training_args.weight_decay},
            {'params': self.f_dc_net.parameters(), 'lr': training_args.feature_lr, 'name': 'f_dc_net', 'weight_decay': training_args.weight_decay},
            {'params': self.f_rest_net.parameters(), 'lr': training_args.feature_lr / 20.0, 'name': 'f_rest_net', 'weight_decay': training_args.weight_decay},
            {'params': self.f_dc_offset_net.parameters(), 'lr': training_args.feature_lr, 'name': 'f_dc_offset_net', 'weight_decay': training_args.weight_decay},
            {'params': self.f_rest_offset_net.parameters(), 'lr': training_args.feature_lr / 20.0, 'name': 'f_rest_offset_net', 'weight_decay': training_args.weight_decay},
        ]
        
        if self.learn_lbs_offset:
            l += [{'params': self.mean_offset_offset_net.parameters(), 'lr': training_args.position_lr_init * self.spatial_lr_scale, 'name': 'mean_offset_offset_net', 'weight_decay': training_args.weight_decay}]

        """
        if mano_param_dict is not None:
            if self.learn_mano_param:
                l += mano_param_dict.get_optimizable_params(training_args.mano_param_lr)

            mano_params = mano_param_dict()[0]
            if self.learn_mano_param:
                self.root_pose = mano_params['root_pose']
                self.hand_pose = mano_params['hand_pose']
                self.trans = mano_params['trans']
            else:
                self.root_pose = mano_params['root_pose'].detach()
                self.hand_pose = mano_params['hand_pose'].detach()
                self.trans = mano_params['trans'].detach()
        """
        if self.learn_mano_param:
            l += mano_param_dict.get_optimizable_params(training_args.mano_param_lr)

        l += [{'params': [self.shape_param], 'lr': training_args.mano_param_lr, "name": "mano_shape"}]

        return l

    def training_setup(self, training_args, mano_param_dict):

        l = self.get_param_list_with_setup_prefix(training_args, mano_param_dict)

        self.optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15, weight_decay=training_args.weight_decay)
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
            if param_group["name"] in ["geo_net", "mean_offset_net", 'geo_offset_net', 'mean_offset_offset_net']:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_feature_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["f_dc_net", "f_rest_net", 'f_dc_offset_net', 'f_rest_offset_net']:
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
            if param_group["name"] in ["scale_net", "scale_offset_net"]:
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # save pth. incompatible with original GaussianDreamerPro
    def save_pth(self, path):
        dirname = os.path.dirname(path)
        if dirname != '':
            mkdir_p(dirname)

        triplane = self.triplane.detach()
        #triplane_face = self.triplane_face
        geo_net = self.geo_net.state_dict()
        mean_offset_net = self.mean_offset_net.state_dict()
        scale_net = self.scale_net.state_dict()
        geo_offset_net = self.geo_offset_net.state_dict()
        f_dc_net = self.f_dc_net.state_dict()
        f_rest_net = self.f_rest_net.state_dict()
        f_dc_offset_net = self.f_dc_offset_net.state_dict()
        f_rest_offset_net = self.f_rest_offset_net.state_dict()
        
        root_pose = self.root_pose.detach()
        hand_pose = self.hand_pose.detach()
        trans = self.trans.detach()

        save_dict = {
            'triplane': triplane,
            #'triplane_face': triplane_face,
            'geo_net': geo_net,
            'mean_offset_net': mean_offset_net,
            'scale_net': scale_net,
            'geo_offset_net': geo_offset_net,
            'f_dc_net': f_dc_net,
            'f_rest_net': f_rest_net,
            'f_dc_offset_net': f_dc_offset_net,
            'f_rest_offset_net': f_rest_offset_net,
            'root_pose': root_pose,
            'hand_pose': hand_pose,
            'trans': trans,
        }

        if self.learn_lbs_offset:
            mean_offset_offset_net = self.mean_offset_offset_net.state_dict()
            save_dict['mean_offset_offset_net'] = mean_offset_offset_net
        
        torch.save(save_dict, path)
    
    def load_pth(self, path):
        save_dict = torch.load(path)
        self.triplane = nn.Parameter(save_dict['triplane'].requires_grad_(True))
        #self.triplane_face = nn.Parameter(save_dict['triplane_face'].requires_grad_(True))
        self.geo_net.load_state_dict(save_dict['geo_net'])
        self.mean_offset_net.load_state_dict(save_dict['mean_offset_net'])
        self.scale_net.load_state_dict(save_dict['scale_net'])
        self.geo_offset_net.load_state_dict(save_dict['geo_offset_net'])
        self.f_dc_net.load_state_dict(save_dict['f_dc_net'])
        self.f_rest_net.load_state_dict(save_dict['f_rest_net'])
        self.f_dc_offset_net.load_state_dict(save_dict['f_dc_offset_net'])
        self.f_rest_offset_net.load_state_dict(save_dict['f_rest_offset_net'])
        
        learn_mano_param = self.learn_mano_param
        self.root_pose = nn.Parameter(save_dict['root_pose'].requires_grad_(learn_mano_param))
        self.hand_pose = nn.Parameter(save_dict['hand_pose'].requires_grad_(learn_mano_param))
        self.trans = nn.Parameter(save_dict['trans'].requires_grad_(learn_mano_param))

    """
    # obsolete
    def save_ply(self, path):
        self.save_pth(path)
    
    # obsolete
    def load_ply(self, path):
        self.load_pth(path)
    """

    def get_lap_loss(self, loss_fn, assets, assets_refined, offsets, mesh_neutral_pose):
        # lap_mean
        """
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_hand_pose = mano.neutral_hand_pose.view(1,-1).cuda() # tempalte pose
        shape_param = self.shape_param
        if shape_param.dim()==1:
            shape_param = shape_param[None,:]
        output = self.mano_layer(global_orient=zero_pose, hand_pose=neutral_hand_pose, betas=shape_param, transl=self.hand_trans)
        mesh_neutral_pose = output.vertices[0]
        mesh_neutral_pose = mano.upsample_mesh(mesh_neutral_pose)
        """
        mesh_neutral_pose = mesh_neutral_pose[None,:,:] # [1, 49281, 3]
        
        #mean_offset = self.mean_offset
        mean_offset = offsets['mean_offset'][None,:,:] # [1, 49281, 3]

        mean_loss = loss_fn(mesh_neutral_pose.detach() + mean_offset, mesh_neutral_pose.detach()).mean()
        if self.learn_lbs_offset:
            #mean_offset_offset = self.mean_offset_offset
            mean_offset_offset = offsets['mean_offset_offset'][None,:,:]
            mean_loss = mean_loss + loss_fn(mesh_neutral_pose.detach() + mean_offset + mean_offset_offset, mesh_neutral_pose.detach()).mean() * 100000
        """
        # lap_scale
        scale = self.scaling_activation(self._scaling).unsqueeze(0)
        scale_refined = self.get_scaling.unsqueeze(0)
        weight = 0.1

        scale_loss = loss_fn(scale, None).mean() + loss_fn(scale_refined, None).mean() * weight

        # lap_rgb. but we use shs. not sure whether this is correct
        shs = torch.cat((self._features_dc, self._features_rest), dim=1).permute(1,0,2)
        shs_refined = self.get_features.permute(1,0,2)
        weight = 0.1
        rgb_loss = loss_fn(shs, None).mean() + loss_fn(shs_refined, None).mean() * weight

        loss = mean_loss + scale_loss + rgb_loss
        """
        loss = mean_loss
        
        return loss
