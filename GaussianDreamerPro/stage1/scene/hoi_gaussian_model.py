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
import pickle
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import open3d as o3d
import random

class HOIGaussianModel:

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


    def __init__(self, sh_degree, obj_gaussians, hand_gaussians):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.obj_rel_trans = torch.empty(0)
        self.obj_rel_rot = torch.empty(0)
        self.obj_rescaling = torch.empty(0)
        
        self.hand_rel_trans = torch.empty(0)
        #self.hand_rel_rot = torch.empty(0)
        self.hand_rescaling = torch.empty(0)

        self.obj_centric = False
        self.global_rot = torch.empty(0)

        self.optimizer = None
        self.spatial_lr_scale = 0

        self.obj_gaussians = obj_gaussians
        self.hand_gaussians = hand_gaussians
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self.obj_rel_trans,
            self.obj_rel_rot,
            self.obj_rescaling,
            self.hand_rel_trans,
            #self.hand_rel_rot,
            self.hand_rescaling,
            self.obj_centric,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args, mano_param_dict=None):
        len_model_args = len(model_args)
        if len_model_args == 10:
            (self.active_sh_degree, 
            self.obj_rel_trans,
            self.obj_rel_rot,
            self.obj_rescaling,
            self.hand_rel_trans,
            _,
            self.hand_rescaling,
            self.obj_centric,
            opt_dict,
            self.spatial_lr_scale) = model_args
        elif len_model_args==9 : # 9
            (self.active_sh_degree, 
            self.obj_rel_trans,
            self.obj_rel_rot,
            self.obj_rescaling,
            self.hand_rel_trans,
            #self.hand_rel_rot,
            self.hand_rescaling,
            self.obj_centric,
            opt_dict,
            self.spatial_lr_scale) = model_args
        elif len_model_args==6:
            (self.active_sh_degree, 
            self.obj_rel_trans,
            self.obj_rel_rot,
            self.obj_rescaling,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        if training_args is not None:
            self.training_setup(training_args, mano_param_dict)
        if self.optimizer is not None and opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_obj_rescaling(self): # rescale obj rescaling factor based on hand rescaling factor (dirty, I admit)
        new_obj_rescaling = self.obj_rescaling * self.hand_gaussians.hand_rescaling
        one_err = (1 - new_obj_rescaling) ** 2
        if one_err < 1.0e-8:
            new_obj_rescaling = 1.0
        return new_obj_rescaling

    @property
    def get_hand_rescaling(self): # rescale obj rescaling factor based on hand rescaling factor (dirty, I admit)
        new_hand_rescaling = self.hand_rescaling
        one_err = (1 - new_hand_rescaling) ** 2
        if one_err < 1.0e-8:
            new_hand_rescaling = 1.0
        return new_hand_rescaling

    @property
    def get_scaling(self):
        if self.obj_centric:
            obj_scaling = self.obj_gaussians.get_scaling
            hand_scaling = self.hand_gaussians.get_scaling * self.get_hand_rescaling
        else:
            obj_scaling = self.obj_gaussians.get_scaling * self.get_obj_rescaling
            hand_scaling = self.hand_gaussians.get_scaling
        hoi_scaling = torch.cat((obj_scaling, hand_scaling), dim=0)
        return hoi_scaling
    
    @property
    def get_rotation(self):
        obj_rot = self.obj_gaussians.get_rotation
        hand_rot = self.hand_gaussians.get_rotation

        if self.obj_centric:
            #hand_rotmat = quaternion_to_matrix(hand_rot)
            #hand_rot = matrix_to_quaternion(hand_rotmat @ self.hand_rel_rot.T)
            obj_rotmat = quaternion_to_matrix(obj_rot)
            obj_rot = matrix_to_quaternion(obj_rotmat @ self.obj_rel_rot.T)
        else:
            obj_rotmat = quaternion_to_matrix(obj_rot)
            obj_rot = matrix_to_quaternion(obj_rotmat @ self.obj_rel_rot.T)

        hoi_rot = torch.cat((obj_rot, hand_rot), dim=0)

        # global rot
        if self.global_rot.shape[0] > 0:
            hoi_rotmat = quaternion_to_matrix(hoi_rot)
            hoi_rot = matrix_to_quaternion(hoi_rotmat @ self.global_rot.T)
        return hoi_rot
    
    @property
    def get_xyz(self):
        obj_xyz = self.obj_gaussians.get_xyz
        hand_xyz = self.hand_gaussians.get_xyz
        if self.obj_centric:
            obj_xyz = obj_xyz @ self.obj_rel_rot.T
            #hand_xyz = (hand_xyz * self.get_hand_rescaling) @ self.hand_rel_rot.T + self.hand_rel_trans
            hand_xyz = (hand_xyz * self.get_hand_rescaling) + self.hand_rel_trans
        else:
            obj_xyz = (obj_xyz * self.get_obj_rescaling) @ self.obj_rel_rot.T + self.obj_rel_trans
        hoi_xyz = torch.cat((obj_xyz, hand_xyz), dim=0)
        
        # global rot
        if self.global_rot.shape[0] > 0:
            hoi_xyz = hoi_xyz @ self.global_rot.T
        return hoi_xyz

    @property
    def get_coarse_xyz(self): # get transformed coarse obj vertex locations + hand GS locations
        obj_xyz = self.obj_gaussians.get_coarse_xyz
        hand_xyz = self.hand_gaussians.get_xyz
        if self.obj_centric:
            obj_xyz =  obj_xyz @ self.obj_rel_rot.T
            #hand_xyz = (hand_xyz * self.get_hand_rescaling) @ self.hand_rel_rot.T + self.hand_rel_trans
            hand_xyz = (hand_xyz * self.get_hand_rescaling) + self.hand_rel_trans
        else:
            obj_xyz = (obj_xyz * self.get_obj_rescaling) @ self.obj_rel_rot.T + self.obj_rel_trans
        hoi_xyz = torch.cat((obj_xyz, hand_xyz), dim=0)

        # global rot
        if self.global_rot.shape[0] > 0:
            hoi_xyz = hoi_xyz @ self.global_rot.T
        return hoi_xyz

    @property
    def get_features(self):
        obj_features = self.obj_gaussians.get_features
        hand_features = self.hand_gaussians.get_features
        hoi_features = torch.cat((obj_features, hand_features), dim=0)
        return hoi_features
    
    @property
    def get_opacity(self):
        obj_opacity = self.obj_gaussians.get_opacity
        hand_opacity = self.hand_gaussians.get_opacity
        hoi_opacity = torch.cat((obj_opacity, hand_opacity), dim=0)
        return hoi_opacity
    
    def get_covariance(self, scaling_modifier = 1):
        if self.obj_centric:
            obj_cov = self.obj_gaussians.get_covariance(scaling_modifier)
            hand_cov = self.hand_gaussians.get_covariance(scaling_modifier * self.get_hand_rescaling)
        else:
            obj_cov = self.obj_gaussians.get_covariance(scaling_modifier * self.get_obj_rescaling)
            hand_cov = self.hand_gaussians.get_covariance(scaling_modifier)
        hoi_cov = torch.cat((obj_cov, hand_cov), dim=0)
        return hoi_cov

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def random_quaternion(self, size):  
        u = torch.rand(size, device="cuda")  
        v = torch.rand(size, device="cuda")  
        w = torch.rand(size, device="cuda")  
        
        sin_2pi_v = torch.sin(2 * np.pi * v)  
        cos_2pi_v = torch.cos(2 * np.pi * v)  
        sin_2pi_w = torch.sin(2 * np.pi * w)  
        cos_2pi_w = torch.cos(2 * np.pi * w)  
        
        q = torch.stack([  
            torch.sqrt(1 - u) * sin_2pi_v,  
            torch.sqrt(1 - u) * cos_2pi_v,  
            torch.sqrt(u) * sin_2pi_w,  
            torch.sqrt(u) * cos_2pi_w  
        ], dim=1)  
        
        return q  

    def create_from_text2hoi(self, t2hoi_data, spatial_lr_scale, 
                            obj_centric=False, obj_random_rescale=False):
        t2hoi_hand_trans = t2hoi_data['hand_trans'] # translation in text2hoi inference result
        if obj_random_rescale:
            object_rescaling_factor = 1.0 - random.random() * 0.2
            obj_rescaling = t2hoi_data['obj_rescaling'] * object_rescaling_factor
        else:
            obj_rescaling = t2hoi_data['obj_rescaling']
        #obj_affine = t2hoi_data['obj_affine']
        #obj_rot = obj_affine[:3,:3]
        #obj_trans = obj_affine[:,3]
        obj_rot = t2hoi_data['obj_rotmat']
        obj_trans = t2hoi_data['obj_trans']
        obj_pc_contact_rhand = t2hoi_data['obj_pc_contact_rhand']
        rhand_contact_joint = t2hoi_data['rhand_contact_joint']
        
        hand_gs_trans = self.hand_gaussians.hand_trans[0] # translation for recentering hand gaussians
        hand_rescaling = self.hand_gaussians.hand_rescaling
        obj_rel_trans = (obj_trans - t2hoi_hand_trans) * hand_rescaling + hand_gs_trans.cpu().numpy()
        hand_rel_trans = (t2hoi_hand_trans - obj_trans - (hand_gs_trans.cpu().numpy() * (1/hand_rescaling))) * (1/obj_rescaling)

        #print(f'hand_gs_trans: {hand_gs_trans}, obj_rel_trans: {obj_rel_trans}')
        #print(f'obj_trans: {obj_trans}, t2hoi_hand_trans: {t2hoi_hand_trans}, hand_gs_trans: {hand_gs_trans}')
        
        device = 'cuda'
        obj_rot_th = torch.tensor(obj_rot, dtype=torch.float32, device=device)
        obj_rel_trans_th = torch.tensor(obj_rel_trans, dtype=torch.float32, device=device).unsqueeze(0)
        obj_rescaling = torch.tensor(obj_rescaling, dtype=torch.float32, device=device)

        self.obj_rel_trans = nn.Parameter(obj_rel_trans_th.requires_grad_(False))
        self.obj_rel_rot = nn.Parameter(obj_rot_th.requires_grad_(False))
        self.obj_rescaling = nn.Parameter(obj_rescaling.requires_grad_(False))

        hand_rel_trans_th = torch.tensor(hand_rel_trans, dtype=torch.float32, device=device).unsqueeze(0)
        hand_rot_th = torch.tensor(obj_rot.T, dtype=torch.float32, device=device)
        hand_rel_rescaling = 1.0/torch.clamp(obj_rescaling * hand_rescaling, 1.0e-8)
        print(f'obj_rescaling: {obj_rescaling}, hand_rescaling: {hand_rescaling}, hand_rel_rescaling: {hand_rel_rescaling}')
        self.hand_rel_trans = nn.Parameter(hand_rel_trans_th.requires_grad_(False))
        #self.hand_rel_rot = nn.Parameter(hand_rot_th.requires_grad_(False))
        self.hand_rescaling = nn.Parameter(hand_rel_rescaling.requires_grad_(False))

        self.obj_centric = obj_centric

        self.obj_pc_contact_rhand = torch.tensor(obj_pc_contact_rhand, dtype=torch.float32, device=device)
        self.rhand_contact_joint = torch.tensor(rhand_contact_joint, dtype=torch.bool, device=device)

        self.global_rot = torch.empty(0)
        if 'global_rot' in t2hoi_data.keys():
            self.global_rot = torch.tensor(t2hoi_data['global_rot'], dtype=torch.float32, device=device)

    def transform_contact_point(self):
        if self.obj_centric:
            #obj_pc_contact_rhand = self.obj_pc_contact_rhand
            obj_pc_contact_rhand = self.obj_pc_contact_rhand @ self.obj_rel_rot.T
        else:
            obj_pc_contact_rhand = self.obj_pc_contact_rhand @ self.obj_rel_rot.T + self.obj_rel_trans
        return obj_pc_contact_rhand

    def training_setup(self, training_args, mano_param_dict):
        # get params of obj and hand gaussians
        obj_list = self.obj_gaussians.get_param_list_with_setup_prefix(training_args)
        hand_list = self.hand_gaussians.get_param_list_with_setup_prefix(training_args, mano_param_dict)

        optim_list = obj_list + hand_list
        self.optimizer = torch.optim.Adam(optim_list, lr=0.0, eps=1e-15)
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
            if param_group["name"] in ["xyz", "mean_offset", "mean_offset_offset"]:
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
            if param_group["name"] in ["scaling", 'scale_offset']:
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
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_pkl(self, path):
        mkdir_p(os.path.dirname(path))

        save_dict = {
            'obj_rel_trans': self.obj_rel_trans.detach().cpu().numpy(),
            'obj_rel_rot': self.obj_rel_rot.detach().cpu().numpy(),
            'obj_rescaling': self.obj_rescaling.detach().cpu().numpy()
        }
        if self.global_rot.shape[0] > 0:
            save_dict['global_rot'] = self.global_rot.detach().cpu().numpy()
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)


    def reset_opacity(self, min_opacity=0.01):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*min_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def clamp_obj_opacity(self, min_opacity=0.5):
        opacities_new = torch.max(self.obj_gaussians._opacity, torch.ones_like(self.obj_gaussians._opacity)*min_opacity)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.obj_gaussians._opacity = optimizable_tensors["opacity"]

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj_rel_trans = data['obj_rel_trans']
        obj_rel_rot = data['obj_rel_rot']
        obj_rescaling = data['obj_rescaling']

        self.obj_rel_trans = nn.Parameter(torch.tensor(obj_rel_trans, dtype=torch.float, device="cuda").requires_grad_(False))
        self.obj_rel_rot = nn.Parameter(torch.tensor(obj_rel_rot, dtype=torch.float, device="cuda").requires_grad_(False))
        self.obj_rescaling = nn.Parameter(torch.tensor(obj_rescaling, dtype=torch.float, device="cuda").requires_grad_(False))

        if 'global_rot' in data.keys():
            self.global_rot = nn.Parameter(torch.tensor(data['global_rot'], dtype=torch.float, device="cuda").requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def meshify(self, depth, mesh_path, target_num_pt, min_opacity=0.9):
        # initialize o3d instance
        pcd = o3d.geometry.PointCloud()
        n_gs = self._xyz.shape[0]
        pcd.points = o3d.utility.Vector3dVector(self._xyz.detach().cpu().numpy())

        rgb = SH2RGB(self.get_features[:,0].detach().float())
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((n_gs,3)))
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(10)

        # outliers removal
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
        n_remove = self._xyz.shape[0] - len(ind)
        
        if n_remove > 0:
            pcd = pcd.select_by_index(ind)
            print(f"remove {n_remove} outlier points")
        

        # mesh recon
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        
        # upsample if needed
        n_v = len(mesh.vertices)
        while (n_v < target_num_pt):
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            n_v = len(mesh.vertices)
        
        # decimate around target_num_pt
        # simplify_quadric_decimation actually decimates to target number of triangles, not vertices. 
        if n_v > target_num_pt:
            mesh = mesh.simplify_quadric_decimation(target_num_pt*2)

        # save mesh
        o3d.io.write_triangle_mesh(mesh_path, mesh, write_triangle_uvs=True, write_vertex_colors=False, write_vertex_normals=True)
        print(f"mesh saved at {mesh_path}")

        # get new gaussian attributes
        print(f'len(mesh.vertices): {len(mesh.vertices)}, len(mesh.triangles): {len(mesh.triangles)}')
    
        new_xyz = torch.tensor(np.array(mesh.vertices), dtype=torch.float, device='cuda').requires_grad_(True)
        new_features_dc = RGB2SH(torch.tensor(np.asarray(mesh.vertex_colors), dtype=torch.float, device='cuda')).unsqueeze(1)
        new_features_rest = torch.zeros((new_features_dc.shape[0], (self.max_sh_degree + 1) ** 2-1, 3), dtype=torch.float, device='cuda')

        dist2 = torch.clamp_min(distCUDA2(new_xyz), 0.0000001)
        new_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        new_rots = self.random_quaternion(new_xyz.shape[0]) 

        new_opacity = min_opacity * torch.ones((new_xyz.shape[0], 1), dtype=torch.float, device="cuda")

        # set optimizer state
        tensors_dict = {"xyz": new_xyz,
                        "f_dc": new_features_dc,
                        "f_rest": new_features_rest,
                        "opacity": new_opacity,
                        "scaling" : new_scales,
                        "rotation" : new_rots}
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.zeros_like(extension_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(extension_tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(extension_tensor).requires_grad_(True)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(extension_tensor).requires_grad_(True)
                optimizable_tensors[group["name"]] = group["params"][0]
        
        # set attributes
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
