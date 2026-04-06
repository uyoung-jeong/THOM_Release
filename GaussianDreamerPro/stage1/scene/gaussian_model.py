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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import open3d as o3d

class GaussianModel:

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
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            optim_dict,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        if training_args is not None:
            self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if self.optimizer is not None and opt_dict is not None:
            print('load object optimizer')
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        scaling = self._scaling
        if self.isotropic:
            scaling = scaling.repeat(1,3)
        return self.scaling_activation(scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_coarse_xyz(self): # to correctly call this, need to call load_coarse_mesh() first
        return self.coarse_verts

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

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

    # fully use lgm inference
    def create_from_lgm(self, plydata, spatial_lr_scale, min_opacity):
        self.spatial_lr_scale = spatial_lr_scale
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = torch.tensor(opacities).cuda()
        
        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
        shs = torch.tensor(shs).cuda()

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        scales = torch.tensor(scales).float().cuda()

        if self.isotropic:
            scales = scales.mean(dim=1)[...,None]
        else:
            scales = scales

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rots = torch.tensor(rots).float().cuda()
        
        # rescale
        pcd_lengths = xyz.max(axis=0) - xyz.min(axis=0)
        pcd_max_len = pcd_lengths.max()
        pcd_zlen = xyz[:,2].max() - xyz[:,2].min()
        max_thr = 1.6
        min_thr = 0.7
        z_thr = 1.6
        pcd_oversize = (pcd_max_len>max_thr).sum()
        pcd_undersize = (pcd_max_len<min_thr).sum()
        z_oversize = (pcd_zlen>z_thr).sum()
        rescaler = 1.0
        if z_oversize > 0:
            rescaler = z_thr / pcd_zlen
            xyz = rescaler * xyz
        elif pcd_oversize > 0:
            rescaler = max_thr / pcd_max_len
            xyz = rescaler * xyz
        elif pcd_undersize > 0:
            rescaler = min_thr / pcd_max_len
            xyz = rescaler * xyz
        xyz = torch.tensor(xyz).float().cuda()
        scales = rescaler * scales

        features = torch.zeros((shs.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = shs
        features[:, 3:, 1:] = 0.0

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, min_opacity=0.9, init_lgm_path=''):
        """
        if init_lgm_path != '':
            plydata = plydata = PlyData.read(init_lgm_path)
            self.create_from_lgm(plydata, spatial_lr_scale, min_opacity)
            return
        """

        self.spatial_lr_scale = spatial_lr_scale
        pcd_points = np.asarray(pcd.points)
        # recenter points
        pcd_points = pcd_points - pcd_points.mean(axis=0, keepdims=True)

        # rescale
        pcd_lengths = pcd_points.max(axis=0) - pcd_points.min(axis=0)
        pcd_max_len = pcd_lengths.max()
        pcd_zlen = pcd_points[:,2].max() - pcd_points[:,2].min()
        max_thr = 1.55
        min_thr = 0.8
        z_thr = 1.55
        pcd_oversize = (pcd_max_len>max_thr).sum()
        pcd_undersize = (pcd_max_len<min_thr).sum()
        z_oversize = (pcd_zlen>z_thr).sum()
        if z_oversize > 0:
            rescaler = z_thr / pcd_zlen
            pcd_points = rescaler * pcd_points
        elif pcd_oversize > 0:
            rescaler = max_thr / pcd_max_len
            pcd_points = rescaler * pcd_points
        elif pcd_undersize > 0:
            rescaler = min_thr / pcd_max_len
            pcd_points = rescaler * pcd_points
        
        fused_point_cloud = torch.tensor(pcd_points).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        #print(f'pcd.colors.min(): {pcd.colors.min()}, pcd.colors.max(): {pcd.colors.max()}') # 0.498
        #print(f'fused_color.min(): {fused_color.min()}, fused_color.max(): {fused_color.max()}') # -0.007

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        #print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        #print(f'fused_point_cloud.amin(dim=0): {fused_point_cloud.amin(dim=0)}, fused_point_cloud.amax(dim=0): {fused_point_cloud.amax(dim=0)}')

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(pcd_points).float().cuda()), 0.0000001)

        if self.isotropic:
            scales = torch.log(torch.sqrt(dist2))[...,None]
        else:
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = self.random_quaternion(fused_point_cloud.shape[0]) 
        # rots[:, 0] = 1

        # import pdb;pdb.set_trace()
        #opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        opacities = min_opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def get_param_list_with_setup_prefix(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        return l

    def training_setup(self, training_args):
        l = self.get_param_list_with_setup_prefix(training_args)

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
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_feature_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "f_dc":
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
            if param_group["name"] == "scaling":
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

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        if hasattr(self, 'faces'):
            if self.faces is not None:
                n_face = self.faces.shape[0]
            else:
                n_face = 0
            print(f'n_face: {n_face}')
        else:
            n_face = 0
        if hasattr(self, 'vert_norms') and n_face>0:
            verts = self.verts.detach().cpu().numpy()
            faces = self.faces.detach().cpu().numpy()
            verts_o3d = o3d.utility.Vector3dVector(verts)
            faces_o3d = o3d.utility.Vector3iVector(faces)
            
            mesh = o3d.geometry.TriangleMesh(verts_o3d, faces_o3d)
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        if hasattr(self, 'vert_norms') and n_face>0:
            print('saving obj mesh info')
            face_element = np.empty(faces.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
            face_element['vertex_indices'] = list(map(tuple, faces))
            fl = PlyElement.describe(face_element, 'face')
            PlyData([el,fl]).write(path)
        else:
            PlyData([el]).write(path)

    def reset_opacity(self, min_opacity=0.01):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*min_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def clamp_opacity(self, min_opacity=0.5):
        opacities_new = torch.max(self._opacity, torch.ones_like(self._opacity)*min_opacity)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        #print(f'features_dc.min(): {features_dc.min()}, features_dc.max(): {features_dc.max()}')

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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.faces = None
        if len(plydata.elements) == 2:
            if plydata.elements[1].name == 'face':
                print('loading obj mesh info')
                faces = np.stack(plydata.elements[1]['vertex_indices'])
                self.faces = torch.from_numpy(faces).to('cuda').requires_grad_(False)

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

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:3], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # extract mesh and keep the gaussians
    def extract_poisson(self, depth, mesh_path, target_num_pt, min_opacity=0.9):
        # initialize o3d instance
        pcd = o3d.geometry.PointCloud()
        n_gs = self._xyz.shape[0]
        pcd.points = o3d.utility.Vector3dVector(self._xyz.detach().cpu().numpy())

        rgb = SH2RGB(self.get_features[:,0].detach().float())
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((n_gs,3)))
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)

        # outliers removal
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
        n_remove = self._xyz.shape[0] - len(ind)
        
        if n_remove > 0:
            pcd = pcd.select_by_index(ind)
            print(f"remove {n_remove} outlier points")
        
        # mesh recon
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        
        # save mesh
        o3d.io.write_triangle_mesh(mesh_path, mesh, write_triangle_uvs=True, write_vertex_colors=False, write_vertex_normals=True)
        print(f"mesh saved at {mesh_path}")

        # freeze xyz
        self._xyz.requires_grad = False
        
        
    # extract mesh and reinitialize gaussians
    def meshify_poisson(self, depth, mesh_path, target_num_pt, min_opacity=0.9):
        # initialize o3d instance
        pcd = o3d.geometry.PointCloud()
        n_gs = self._xyz.shape[0]
        pcd.points = o3d.utility.Vector3dVector(self._xyz.detach().cpu().numpy())

        rgb = SH2RGB(self.get_features[:,0].detach().float())
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((n_gs,3)))
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)

        # outliers removal
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
        n_remove = self._xyz.shape[0] - len(ind)
        
        if n_remove > 0:
            pcd = pcd.select_by_index(ind)
            print(f"remove {n_remove} outlier points")

        # mesh recon
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

        # clean mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # upsample if needed
        n_v = len(mesh.vertices)
        while (n_v < int(target_num_pt*0.6)):
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            n_v = len(mesh.vertices)
        
        # decimate around target_num_pt
        # simplify_quadric_decimation actually decimates to target number of triangles, not vertices. 
        """
        if n_v > target_num_pt:
            mesh = mesh.simplify_quadric_decimation(target_num_pt*2)
        """

        # recompute vertex normal
        mesh.compute_vertex_normals()

        # save mesh
        o3d.io.write_triangle_mesh(mesh_path, mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        print(f"mesh saved at {mesh_path}")

        # get new gaussian attributes
        print(f'len(mesh.vertices): {len(mesh.vertices)}, len(mesh.triangles): {len(mesh.triangles)}')

        device = 'cuda'
        # get verts, faces, normals
        self.verts = torch.from_numpy(np.asarray(mesh.vertices)).to(device).requires_grad_(False)
        self.faces = torch.from_numpy(np.asarray(mesh.triangles)).to(device).requires_grad_(False)
        self.vert_norms = torch.from_numpy(np.asarray(mesh.vertex_normals)).to(device).requires_grad_(False)

        new_xyz = torch.tensor(np.array(mesh.vertices), dtype=torch.float, device=device).requires_grad_(True)
        new_features_dc = RGB2SH(torch.tensor(np.asarray(mesh.vertex_colors), dtype=torch.float, device='cuda')).unsqueeze(1)
        new_features_rest = torch.zeros((new_features_dc.shape[0], (self.max_sh_degree + 1) ** 2-1, 3), dtype=torch.float, device='cuda')

        dist2 = torch.clamp_min(distCUDA2(new_xyz), 0.0000001)
        if self.isotropic:
            new_scales = torch.log(torch.sqrt(dist2))[...,None]
        else:
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

    def meshify_ball_pivot(self, radii, mesh_path, target_num_pt, min_opacity=0.9, remove_outlier=True):
        # initialize o3d instance
        pcd = o3d.geometry.PointCloud()
        n_gs = self._xyz.shape[0]
        pcd.points = o3d.utility.Vector3dVector(self._xyz.detach().cpu().numpy())

        rgb = SH2RGB(self.get_features[:,0].detach().float())
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((n_gs,3)))
        
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)

        # outliers removal
        if remove_outlier:
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
            n_remove = self._xyz.shape[0] - len(ind)
            if n_remove > 0:
                pcd = pcd.select_by_index(ind)
                print(f"remove {n_remove} outlier points")
        else:
            n_remove = 0
        
        # mesh recon
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        n_v = len(mesh.vertices)
        reinit_feats = False
        if n_v != n_gs-n_remove: # if this happens, we choose to lose all learned features, since we cannot find correspondence
            print(f'number of vertices has been changed during mesh reconstruction process.')
            reinit_feats = True

        # upsample if needed
        """
        while (n_v < target_num_pt):
            print(f"upsampling {n_v} vertices")
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            n_v = len(mesh.vertices)
        """
        
        # decimate around target_num_pt
        # simplify_quadric_decimation actually decimates to target number of triangles, not vertices. 
        """
        if n_v > target_num_pt:
            print(f'decimating {n_v} vertices to {target_num_pt}')
            mesh = mesh.simplify_quadric_decimation(target_num_pt*2)
        """

        # save mesh
        o3d.io.write_triangle_mesh(mesh_path, mesh, write_triangle_uvs=True, write_vertex_colors=False, write_vertex_normals=True)
        print(f"mesh saved at {mesh_path}")

        # get new gaussian attributes
        print(f'len(mesh.vertices): {len(mesh.vertices)}, len(mesh.triangles): {len(mesh.triangles)}')
        
        if reinit_feats: # abandon all learned features and restart :(
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
        elif n_gs == n_v: # no need to reassign parameters
            return
        else:
            new_xyz = self._xyz[ind]
            new_features_dc = self._features_dc[ind]
            new_features_rest = self._features_rest[ind]
            new_scales = self._scaling[ind]
            new_rots = self._rotation[ind]
            new_opacity = self._opacity[ind]

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

                    stored_state["exp_avg"] = stored_state["exp_avg"][ind]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][ind]

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

    # extract mesh and reinitialize gaussians
    def initialize_from_sugar_mesh(self, mesh_path, min_opacity=0.9):
        # load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        device = 'cuda'
        # get verts, faces, normals
        self.verts = torch.from_numpy(np.asarray(mesh.vertices)).to(device).requires_grad_(False)
        self.faces = torch.from_numpy(np.asarray(mesh.triangles)).to(device).requires_grad_(False)
        self.vert_norms = torch.from_numpy(np.asarray(mesh.vertex_normals)).to(device).requires_grad_(False)

        new_xyz = torch.tensor(np.array(mesh.vertices), dtype=torch.float, device=device).requires_grad_(True)
        new_features_dc = RGB2SH(torch.tensor(np.asarray(mesh.vertex_colors), dtype=torch.float, device='cuda')).unsqueeze(1)
        new_features_rest = torch.zeros((new_features_dc.shape[0], (self.max_sh_degree + 1) ** 2-1, 3), dtype=torch.float, device='cuda')

        dist2 = torch.clamp_min(distCUDA2(new_xyz), 0.0000001)
        if self.isotropic:
            new_scales = torch.log(torch.sqrt(dist2))[...,None]
        else:
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

    def load_mesh(self, mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        device = 'cuda'
        # get verts, faces, normals
        self.verts = torch.from_numpy(np.asarray(mesh.vertices)).to(device).requires_grad_(False)
        self.faces = torch.from_numpy(np.asarray(mesh.triangles)).to(device).requires_grad_(False)
        self.vert_norms = torch.from_numpy(np.asarray(mesh.vertex_normals)).to(device).requires_grad_(False)

    def load_coarse_mesh(self, mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        device = 'cuda'
        # get verts, faces, normals
        self.coarse_verts = torch.from_numpy(np.asarray(mesh.vertices)).to(device).requires_grad_(False).float()
        self.coarse_faces = torch.from_numpy(np.asarray(mesh.triangles)).to(device).requires_grad_(False)
        self.coarse_vert_norms = torch.from_numpy(np.asarray(mesh.vertex_normals)).to(device).requires_grad_(False).float()

    def get_lap_loss(self, loss_fn):
        # lap_xyz
        verts = self.verts.unsqueeze(0)
        xyz = self.get_xyz
        xyz = xyz.unsqueeze(0)
        mean_loss = loss_fn(verts.detach(), xyz).mean() * 100000
        
        # lap_scale
        scale = self.get_scaling.unsqueeze(0)
        weight = 100000
        scale_loss = loss_fn(scale, None).mean() * weight

        # lap_rgb
        rgb = SH2RGB(self.get_features[:,0].float()).unsqueeze(0)
        weight = 10
        rgb_loss = loss_fn(rgb, None).mean() * weight

        loss = mean_loss + scale_loss + rgb_loss
        
        loss = mean_loss
        
        return loss

    # https://github.com/j-alex-hanson/speedy-splat/blob/speedy-splat/scene/gaussian_model.py
    def prune_gaussians(self, percent, import_score: list):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)
