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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn

import numpy as np
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_points
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
from simple_knn._C import distCUDA2

import lpips
import math

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])  
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size    

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def cal_opacity_loss(_opacity, eps=1e-6):
    '''
    opacity \in [0, 1]
    '''
    opacity = _opacity * (1 - 2*eps) + eps
    return torch.mean(-opacity * torch.log2(opacity))
    # return torch.mean(opacity * torch.log2(opacity) + 0.5)

def cal_splat_loss(scales):
    # min_scales = torch.min(scales, dim=-1)[0]
    # return torch.mean(min_scales)
    return torch.mean(scales)

def cal_tv_loss(inputs, losstype='l1', stage=1):
    ''' 
    Returns TV norm for input values.
    inputs: [c, H, W]
    '''
    if losstype == 'pooling':
        kernel_size = 2 * stage + 1
        padding = stage
        with torch.no_grad():
            smoothed_normals = F.pad(inputs, (padding, padding, padding, padding), mode='replicate')
            smoothed_normals = F.avg_pool2d(smoothed_normals, kernel_size=kernel_size, stride=1)
        loss = (inputs - smoothed_normals).abs().mean()
        # loss = ((inputs - smoothed_normals)**2).mean()
    elif losstype == 'l2' or losstype == 'l1':
        step = stage
        v00 = inputs[:, :-step, :-step]
        v01 = inputs[:, :-step, step:]
        v10 = inputs[:, step:, :-step]
        if losstype == 'l2':
            loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2).mean()
        elif losstype == 'l1':
            loss = (torch.abs(v00 - v01) + torch.abs(v00 - v10)).mean()
        else:
            raise ValueError('Not supported losstype.')
    return loss

class NormalLoss:
    def __init__(self, H, W, focal_x, focal_y, device='cuda'):
        cx, cy = 0.5 * W - 0.5, 0.5 * H - 0.5
        Y, X = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        view_dirs = torch.stack([(X-cx)/focal_x, (Y-cy)/focal_y, torch.ones_like(X)], dim=0).float().to(device)
        self.view_dirs = view_dirs / torch.norm(view_dirs, dim=0, keepdim=True) # [3, H, W]

    def grad_operator(self, depth, radius=1): # distance: [1, H, W]
        points = depth * self.view_dirs # [3, H, W]
        points_pad = F.pad(points, (radius, radius, radius, radius), "replicate")
        stride = int(radius * 2)
        grad_x = points_pad[:, :, stride:] - points_pad[:, :, :-stride]
        grad_y = points_pad[:, stride:] - points_pad[:, :-stride]
        depth_normal = torch.cross(grad_x[:, radius:-radius], grad_y[:, :, radius:-radius], dim=0) # [3, H, W]
        depth_normal /= (torch.norm(depth_normal, dim=0, keepdim=True) + 1e-7)
        return depth_normal # [3, H, W]

    def __call__(self, _normals, _depth, accums, stage=1, accum_thres=0.95):
        '''
        normals: [3, H, W]
        depth: [1, H, W]
        '''
        normals = _normals.clone().permute(1,2,0) # [H, W, 3]
        depth = _depth.detach().clone()
        accum = accums.detach().permute(1,2,0)
        valid_mask = (accum[..., 0] > accum_thres)
        if valid_mask.sum() == 0:
            return torch.tensor(0.).to(normals.device)
        normals[valid_mask] /= accum[valid_mask]
        with torch.no_grad():
            depth_normal = self.grad_operator(depth, radius=stage).permute(1,2,0) # [H, W, 3]
        loss = (1. - torch.sum((normals * depth_normal)[valid_mask], dim=-1).abs()).mean()
        return loss

class LaplacianReg(nn.Module):
    def __init__(self, vertex_num, face):
        super(LaplacianReg, self).__init__()
        self.neighbor_idxs, self.neighbor_weights = self.get_neighbor(vertex_num, face)

    def get_neighbor(self, vertex_num, face, neighbor_max_num = 10):
        adj = {i: set() for i in range(vertex_num)}
        for i in range(len(face)):
            for idx in face[i]:
                adj[idx] |= set(face[i]) - set([idx])

        neighbor_idxs = np.tile(np.arange(vertex_num)[:,None], (1, neighbor_max_num))
        neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)
        for idx in range(vertex_num):
            neighbor_num = min(len(adj[idx]), neighbor_max_num)
            neighbor_idxs[idx,:neighbor_num] = np.array(list(adj[idx]))[:neighbor_num]
            if neighbor_num == 0:
                neighbor_weights[idx,:neighbor_num] = 0
            else:
                neighbor_weights[idx,:neighbor_num] = -1.0 / neighbor_num
        
        neighbor_idxs, neighbor_weights = torch.from_numpy(neighbor_idxs).cuda(), torch.from_numpy(neighbor_weights).cuda()
        return neighbor_idxs, neighbor_weights
    
    def compute_laplacian(self, x, neighbor_idxs, neighbor_weights):
        lap = x + (x[:, neighbor_idxs] * neighbor_weights[None, :, :, None]).sum(2)
        #print(f'x.shape: {x.shape}') # [1, 196993, 3]
        #print(f'neighbor_idxs.shape: {neighbor_idxs.shape}') # [196993, 10]
        #print(f'neighbor_weights.shape: {neighbor_weights.shape}') # [196993, 10]
        #xw = (x[:, neighbor_idxs] * neighbor_weights[None, :, :, None]).sum(2)
        #print(f'xw.shape: {xw.shape}') # [1, 196993, 10, 3]
        #print(f'lap.shape: {lap.shape}') # [1, 196993, 3]
        return lap

    def forward(self, out, target):
        if target is None:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            loss = lap_out ** 2
            return loss
        else:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            lap_target = self.compute_laplacian(target, self.neighbor_idxs, self.neighbor_weights)
            loss = (lap_out - lap_target) ** 2
            return loss

def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)

# https://github.com/JunukCha/Text2HOI/blob/main/lib/utils/proc_output.py
def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    '''
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx

def get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
    '''
    :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    '''
    N1, N2 = src_xyz.size(1), trg_xyz.size(1)

    # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
    NN_src_xyz = batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
    NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
    NN_src_normal = batched_index_select(src_face_normal, trg_NN_idx)

    interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false
    return interior

# https://github.com/JunukCha/Text2HOI/blob/main/lib/utils/proc.py
def get_hand2obj_dist(hand_joints, obj_pc, obj_pc_normal):
    B = hand_joints.shape[0]
    hand_joints = hand_joints.reshape(B, -1, 3)
    obj_pc = obj_pc.reshape(B, -1, 3)
    obj_pc_normal = obj_pc_normal.reshape(B, -1, 3)
    hand_nn_dist, hand_nn_idx = get_NN(hand_joints, obj_pc)
    hand_interior = get_interior(obj_pc_normal, obj_pc, hand_joints, hand_nn_idx)
    hand_nn_dist = hand_nn_dist.sqrt()
    # hand_nn_dist[hand_interior] *= -1
    hand_nn_dist = torch.abs(hand_nn_dist)
    hand_nn_idx_expand = hand_nn_idx.unsqueeze(-1).expand(*hand_nn_idx.shape, 3)
    obj_pc_contact = torch.gather(obj_pc, 1, hand_nn_idx_expand)
    hand_dist_values_xyz = (hand_joints-obj_pc_contact)**2
    hand_dist_values_xyz = hand_dist_values_xyz.reshape(B, -1, 3)
    hand_nn_dist = hand_nn_dist.reshape(B, -1)
    obj_pc_contact = obj_pc_contact.reshape(B, -1, 3)
    return hand_dist_values_xyz, hand_nn_dist, obj_pc_contact

class PenetrationLoss(nn.Module):
    def __init__(self, obj_gaussians, hand_gaussians, 
                use_joint_contact, use_obj_contact, use_penet_sum=False, use_erf=False,
                opt_with_coarse=False, use_ik_loss=False):
        super(PenetrationLoss, self).__init__()
        self.opt_with_coarse = opt_with_coarse
        if opt_with_coarse:
            obj_triangles = obj_gaussians.coarse_faces.unsqueeze(0)
            self.obj_triangles = obj_triangles
        else:
            faces = obj_gaussians.faces
            if faces is not None:
                obj_triangles = obj_gaussians.faces.unsqueeze(0)
                self.obj_triangles = obj_triangles
            else:
                self.obj_triangles = None

        hand_triangles = hand_gaussians.mano_layer.faces.astype(int)
        hand_triangles = torch.from_numpy(hand_triangles).unsqueeze(0)

        self.hand_triangles = hand_triangles

        self.use_joint_contact = use_joint_contact
        self.use_obj_contact = use_obj_contact
        self.use_penet_sum = use_penet_sum
        self.use_erf = use_erf
        self.use_ik_loss = use_ik_loss
        # vertex idxs for the palm side of the MANO joints
        self.palm_idxs = [34, 62, 194, 238, 288, 386, 397, 604, 614, 625, 141, 496, 507, 114, 126, 755, 763, 350, 439, 573, 690]

        # vertex idxs for the back side of the MANO joints. We only consider the peripheral joints
        self.back_idxs = [191, 144, 212, 283, 270, 388, 405, 289, 590, 628, 290, 498, 516, 229, 29, 708, 724, 311, 423, 534, 651]
        
    def get_obj_contact_verts(self, hoi_gaussians):
        if self.opt_with_coarse:
            xyz = hoi_gaussians.get_coarse_xyz
            n_obj_gs = hoi_gaussians.obj_gaussians.coarse_verts.shape[0]
            #print(f'xyz.shape: {xyz.shape}, n_obj_gs: {n_obj_gs}')
            #print(f'number of coarse obj verts: {n_obj_gs}, hoi_gaussians.obj_gaussians.coarse_verts.shape: {hoi_gaussians.obj_gaussians.coarse_verts.shape}')
        else:
            xyz = hoi_gaussians.get_xyz
            n_obj_gs = hoi_gaussians.obj_gaussians._xyz.shape[0]
        obj_xyz = xyz[:n_obj_gs]
        hand_xyz = xyz[n_obj_gs:]
        
        #print(f'obj_xyz.mean(dim=0): {obj_xyz.mean(dim=0)}, hand_xyz.mean(dim=0): {hand_xyz.mean(dim=0)}')
        #print(f'xyz.shape: {xyz.shape}, obj_xyz.shape: {obj_xyz.shape}, hand_xyz.shape: {hand_xyz.shape}')

        obj_v_tensor = obj_xyz.unsqueeze(0)
        hand_v_tensor = hand_xyz.unsqueeze(0)

        device = obj_v_tensor.device
        hand_mesh = Meshes(hand_v_tensor, self.hand_triangles.to(device))
        rhand_normal = hand_mesh.verts_normals_packed().view(1, -1, 3)

        # get penetration info
        nn_dists, nn_idx = get_NN(obj_v_tensor, hand_v_tensor)
        interior = get_interior(rhand_normal, hand_v_tensor, obj_v_tensor, nn_idx)

        n_interior = interior.sum().item()
        close_thr = 1.0e-4
        if n_interior > 0:
            close_mask = nn_dists<close_thr
            contact_map = (interior + close_mask)>0
        else:
            nn_dists_sorted, dist_indices = torch.sort(nn_dists[0])
            #print(f'obj_v_tensor.shape: {obj_v_tensor.shape}, hand_xyz.shape: {hand_xyz.shape}')
            #print(f'nn_dists.shape: {nn_dists.shape}, nn_dists_sorted.shape: {nn_dists_sorted.shape}')
            if len(nn_dists_sorted) >= 10:
                close_thr = nn_dists_sorted[10]
            else:
                close_thr = nn_dists_sorted[-1]
            contact_map = nn_dists<close_thr
            contact_map = contact_map
        n_contact = contact_map.sum().item()
        #print(f'n_contact for object contact loss: {n_contact}, contact_map.shape: {contact_map.shape}')
        if n_contact > 0:
            self.obj_contact_map = contact_map # [1,n_obj_v]
        else:
            self.obj_contact_map = None
    
    def get_hand_contact_joints(self, hoi_gaussians):
        rhand_contact_joint = hoi_gaussians.rhand_contact_joint.detach()
        rhand_contact_joint_sum = rhand_contact_joint.sum()
        #print(f'rhand_contact_joint_sum before contact map modification: {rhand_contact_joint_sum}')
        
        reinit_contact = True
        if reinit_contact or (rhand_contact_joint_sum.item() == 0):
            #if rhand_contact_joint_sum.item() == 0:
            # add contact for the joints in proximity of the object
            tip_idxs = [745, 317, 445, 556, 673]
            mano_layer = hoi_gaussians.hand_gaussians.mano_layer
            root_pose = hoi_gaussians.hand_gaussians.root_pose
            if root_pose.shape[-1] == 6: # 6d representation
                root_pose = matrix_to_axis_angle(rotation_6d_to_matrix(root_pose))

            hand_pose = hoi_gaussians.hand_gaussians.hand_pose
            if hand_pose.shape[1] == 6: # 6d representation
                hand_pose = matrix_to_axis_angle(rotation_6d_to_matrix(hand_pose))
            hand_pose = hand_pose.reshape(1, -1)
            hand_shape = hoi_gaussians.hand_gaussians.shape_param
            hand_trans = hoi_gaussians.hand_gaussians.trans + hoi_gaussians.hand_gaussians.hand_trans
            if hand_trans.dim()==2:
                hand_trans = hand_trans.unsqueeze(1)
            mano_out = mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=hand_shape) # might need rescaling

            #print(f'mano_out.joints.shape: {mano_out.joints.shape}, mano_joints_tip.shape: {mano_joints_tip.shape}')
            #mano_joints = mano_joints_tip + hand_trans # [1, 21, 3]
            #mano_joints = mano_out.vertices[0:1,self.palm_idxs] + hand_trans # [1, 21, 3]
            hand_rescaling = hoi_gaussians.hand_gaussians.hand_rescaling
            mano_joints = mano_out.vertices[0:1,self.palm_idxs] * hand_rescaling + hand_trans # [1, 21, 3]

            xyz = hoi_gaussians.get_xyz
            n_obj_gs = hoi_gaussians.obj_gaussians._xyz.shape[0]
            obj_xyz, hand_xyz = xyz[:n_obj_gs], xyz[n_obj_gs:]
            if self.opt_with_coarse:
                coarse_xyz = hoi_gaussians.get_coarse_xyz
                n_coarse_obj_gs = hoi_gaussians.obj_gaussians.coarse_verts.shape[0]
                coarse_obj_xyz = coarse_xyz[:n_coarse_obj_gs]
            obj_v_tensor = obj_xyz.unsqueeze(0)
            if self.opt_with_coarse:
                obj_v_tensor = coarse_obj_xyz.unsqueeze(0)
            device = obj_v_tensor.device
            if self.obj_triangles is None:
                return
            obj_mesh = Meshes(obj_v_tensor, self.obj_triangles.to(device))
            obj_normal = obj_mesh.verts_normals_packed().view(1, -1, 3)
            
            rhand_dist_xyz, rhand_nn_dist, obj_pc_contact_rhand = get_hand2obj_dist(mano_joints, obj_v_tensor, obj_normal)
            #print(f'rhand_nn_dist.shape: {rhand_nn_dist.shape}, rhand_contact_joint.shape: {rhand_contact_joint.shape}')
            nn_dists_sorted, dist_indices = torch.sort(rhand_nn_dist[0])
            close_thr = nn_dists_sorted[6]
            #close_thr = 0.275
            rhand_contact_joint[rhand_nn_dist[0]<close_thr] = True
            hoi_gaussians.rhand_contact_joint = rhand_contact_joint.clone().detach()
            #print(f'rhand_contact_joint_sum after contact map modification: {hoi_gaussians.rhand_contact_joint.sum()}')
            
    def forward(self, hoi_gaussians, return_penetration_only=False, pene_weight=1.0,
                contact_loss_weight=1.0, obj_contact_weight=1.0e-3, ik_loss_weight=1.0, return_dict=False):
        #xyz = hoi_gaussians.get_xyz
        #n_obj_gs = hoi_gaussians.obj_gaussians._xyz.shape[0]
        xyz = hoi_gaussians.get_xyz
        n_obj_gs = hoi_gaussians.obj_gaussians._xyz.shape[0]
        obj_xyz, hand_xyz = xyz[:n_obj_gs], xyz[n_obj_gs:]

        if self.opt_with_coarse:
            coarse_xyz = hoi_gaussians.get_coarse_xyz
            n_coarse_obj_gs = hoi_gaussians.obj_gaussians.coarse_verts.shape[0]
            coarse_obj_xyz = coarse_xyz[:n_coarse_obj_gs]

        # prepare v, mesh, normal
        obj_v_tensor = obj_xyz.unsqueeze(0)
        hand_v_tensor = hand_xyz.unsqueeze(0)
        if torch.isnan(obj_v_tensor).any().item():
            print(f'obj_v_tensor has nan value')
        if torch.isnan(hand_v_tensor).any().item():
            print(f'hand_v_tensor has nan value')
        device = obj_v_tensor.device
        hand_mesh = Meshes(hand_v_tensor, self.hand_triangles.to(device))
        rhand_normal = hand_mesh.verts_normals_packed().view(1, -1, 3)

        # get penetration info
        nn_dists, nn_idx = get_NN(obj_v_tensor, hand_v_tensor)
        interior = get_interior(rhand_normal, hand_v_tensor, obj_v_tensor, nn_idx)
        if torch.isnan(nn_dists).any().item():
            print(f'nn_dists has nan value')

        # minimize general penetration
        nn_dist = nn_dists.sqrt()
        nn_dist_interior = nn_dist[interior] # [n_penetrate]

        ##### Added loss term ######
        sigmoid_alpha = 250 #hyper-param
        sigmoid_weight = 0.1

        if interior.sum() > 0:
            mean_term = nn_dist_interior.mean() # mean-depth penalty

            if self.use_penet_sum:
                pene_sum_loss = sigmoid_weight * torch.clamp(torch.sigmoid(sigmoid_alpha*nn_dist_interior.sum())-0.5, 0.0)
            penet_loss = mean_term
        else:
            penet_loss = torch.zeros(1).cuda().sum()
            if self.use_penet_sum:
                pene_sum_loss = torch.zeros(1).cuda().sum()
        penet_loss = penet_loss * pene_weight
        
        if return_penetration_only:
            return penet_loss

        # contact distance loss
        hand_rescaling = hoi_gaussians.hand_gaussians.hand_rescaling
        use_mano_for_contact = True
        if use_mano_for_contact:
            tip_idxs = [745, 317, 445, 556, 673]
            mano_layer = hoi_gaussians.hand_gaussians.mano_layer
            root_pose = hoi_gaussians.hand_gaussians.root_pose
            if root_pose.shape[-1] == 6: # 6d representation
                root_pose = matrix_to_axis_angle(rotation_6d_to_matrix(root_pose))

            hand_pose = hoi_gaussians.hand_gaussians.hand_pose
            if hand_pose.shape[1] == 6: # 6d representation
                hand_pose = matrix_to_axis_angle(rotation_6d_to_matrix(hand_pose))
            hand_pose = hand_pose.reshape(1, -1)
            hand_shape = hoi_gaussians.hand_gaussians.shape_param
            hand_trans = hoi_gaussians.hand_gaussians.trans + hoi_gaussians.hand_gaussians.hand_trans
            if hand_trans.dim()==2:
                hand_trans = hand_trans.unsqueeze(1)
            mano_out = mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=hand_shape) # might need rescaling

            #print(f'mano_out.joints.shape: {mano_out.joints.shape}, mano_joints_tip.shape: {mano_joints_tip.shape}')
            #mano_joints = mano_joints_tip + hand_trans # [1, 21, 3]
            #mano_joints = mano_out.vertices[0:1,self.palm_idxs] + hand_trans # [1, 21, 3]
            mano_joints = mano_out.vertices[0:1,self.palm_idxs] * hand_rescaling + hand_trans # [1, 21, 3]
        else:
            mano_joints = hand_v_tensor[0:1,self.palm_idxs] # [1, 21, 3]
        
        if self.opt_with_coarse:
            obj_v_tensor = coarse_obj_xyz.unsqueeze(0)
        
        if self.obj_triangles is None:
            contact_loss_avg = contact_loss = torch.zeros(1).cuda().sum()
        else:
            obj_mesh = Meshes(obj_v_tensor, self.obj_triangles.to(device))
            obj_normal = obj_mesh.verts_normals_packed().view(1, -1, 3)
            
            rhand_contact_joint = hoi_gaussians.rhand_contact_joint.detach()

            #print(f'obj_pc_contact_rhand.shape: {obj_pc_contact_rhand.shape}, hand_v_tensor.shape: {hand_v_tensor.shape}') # [1,21,3], [1, 49281,3]
            #print(f'rhand_contact_joint.shape: {rhand_contact_joint.shape}') # [21]
            #print(f'obj_contact_to_handv_nn_idx.shape: {obj_contact_to_handv_nn_idx.shape}') # [1,21]
            
            rhand_dist_xyz, rhand_nn_dist, obj_pc_contact_rhand = get_hand2obj_dist(mano_joints, obj_v_tensor, obj_normal)
            contact_loss = F.mse_loss(mano_joints, obj_pc_contact_rhand, reduction='none').mean(dim=0)
        
            rhand_contact_joint_sum = rhand_contact_joint.sum()
            if rhand_contact_joint_sum.item() == 0:
                #print(f'rhand_nn_dist.shape: {rhand_nn_dist.shape}, rhand_contact_joint.shape: {rhand_contact_joint.shape}')
                nn_dists_sorted, dist_indices = torch.sort(rhand_nn_dist[0])
                close_thr = nn_dists_sorted[5]
                rhand_contact_joint[rhand_nn_dist[0]<close_thr] = True
                hoi_gaussians.rhand_contact_joint = rhand_contact_joint
                
            contact_loss_masked = contact_loss * rhand_contact_joint.unsqueeze(1).float()
            contact_loss_avg = contact_loss_weight * contact_loss_masked.sum() / rhand_contact_joint.sum()
        
        # obj contact map loss
        if self.obj_contact_map is not None:
            # get penetration info using coarse object mesh
            nn_dists, nn_idx = get_NN(obj_v_tensor, hand_v_tensor)
            interior = get_interior(rhand_normal, hand_v_tensor, obj_v_tensor, nn_idx)

            # minimize general penetration
            nn_dist = nn_dists.sqrt()
            obj_contact_map_loss = nn_dist[0,self.obj_contact_map[0]].mean()
        else:
            obj_contact_map_loss = .0
        
        # external-penetration repulsion force
        # https://github.com/4DVLab/DexGrasp-Anything/blob/main/utils/handmodel.py
        if self.use_erf:
            obj_nn_idx_expand = nn_idx.unsqueeze(-1).expand(*nn_idx.shape, 3)
            hand_nn_v = torch.gather(hand_v_tensor, 1, obj_nn_idx_expand)

            hand_obj_signs = ((obj_v_tensor - hand_nn_v + 1e-13) * obj_normal).sum(dim=2)
            hand_obj_signs = (hand_obj_signs > 0.).float()
            collision_value = (hand_obj_signs * nn_dists).amax(dim=1)
            ERF_loss = collision_value.mean()

            loss = penet_loss + ERF_loss
        else:
            loss = penet_loss
        
        if self.use_joint_contact:
            loss = loss + contact_loss_avg
        
        if self.use_obj_contact:
            loss = loss + obj_contact_map_loss * obj_contact_weight

        # penalize back contact
        if self.use_joint_contact and self.obj_triangles is not None:
            back_joints = mano_out.vertices[0:1,self.back_idxs] * hand_rescaling + hand_trans
            rhand_dist_xyz, rhand_nn_dist, obj_pc_contact_rhand_back = get_hand2obj_dist(back_joints, obj_v_tensor, obj_normal)
            back_contact_loss = F.mse_loss(back_joints, obj_pc_contact_rhand_back, reduction='none').mean(dim=0)
            back_contact_loss_masked = back_contact_loss * rhand_contact_joint.unsqueeze(1).float()
            back_reg_loss = torch.clamp(contact_loss_masked - back_contact_loss_masked, .0) * rhand_contact_joint.unsqueeze(1).float()
            #back_reg_loss = contact_loss_weight * back_reg_loss.sum() / rhand_contact_joint.sum()
            back_reg_loss = 5.0 * back_reg_loss.sum() / rhand_contact_joint.sum()
            loss = loss + back_reg_loss
        else:
            back_reg_loss = .0
        
        if self.use_ik_loss and self.obj_triangles is not None:
            ik_target_joints = obj_pc_contact_rhand.detach().clone()
            nn_dists, nn_idx = get_NN(mano_joints, obj_v_tensor)
            interior = get_interior(obj_normal, obj_v_tensor, mano_joints, nn_idx) # [1,21]
            ik_mask = (rhand_contact_joint.unsqueeze(0) + interior)>0
            #ik_mask = interior
            if interior.sum().item() > 0:
                ik_loss = torch.sum(torch.sum((mano_joints - ik_target_joints) ** 2,dim=-1) * ik_mask) / ik_mask.sum()
            else:
                ik_loss = torch.zeros(1).cuda().sum()
            loss = loss + ik_loss

        """
        # non-contact joint contact loss
        hand_nn_dists, hand_nn_idx = get_NN(mano_joints, obj_v_tensor) # hand_nn_idx.shape: [1, 21]
        hand_interior = get_interior(obj_normal, obj_v_tensor, mano_joints, hand_nn_idx)
        hand_interior_mask = hand_interior # [1, 21]
        hand_interior_mask_sum = hand_interior_mask.sum()
        if hand_interior_mask_sum.item() > 0:
            non_contact_joint_loss = F.mse_loss(mano_joints, obj_pc_contact_rhand, reduction='none').mean(dim=0)
            non_contact_joint_loss = non_contact_joint_loss * hand_interior_mask.unsqueeze(2).float()
            non_contact_joint_loss = non_contact_joint_loss.sum() / hand_interior_mask_sum
        else:
            non_contact_joint_loss = .0
        loss = penet_loss + contact_loss_avg * 1.0 + non_contact_joint_loss
        """
        #print(f'loss: {loss}, penet_loss: {penet_loss}')
        if return_dict:
            loss_dict = {'penetration': penet_loss,
                    'hand_contact': contact_loss_avg,
                    'back_reg': back_reg_loss,
                    }
            if self.use_penet_sum:
                loss_dict['pene_sum'] = pene_sum_loss
            if self.use_obj_contact:
                loss_dict['obj_contact'] = obj_contact_map_loss * obj_contact_weight
            if self.use_ik_loss and self.obj_triangles is not None:
                loss_dict['ik'] = ik_loss * ik_loss_weight
            return loss_dict
        else:
            return loss

class GSScaleLoss(nn.Module):
    def __init__(self, ):
        super(GSScaleLoss, self).__init__()

    def forward(self, xyz, scales, isotropic=True):
        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        #base_scales = torch.log(torch.sqrt(dist2))[...,None]
        base_scales = torch.sqrt(dist2)[...,None].detach()
        
        mse_term = F.mse_loss(scales, base_scales)

        max_base_scale = base_scales.amax()
        min_base_scale = base_scales.amin()
        scale_range = max_base_scale - min_base_scale

        overflow_thr = max_base_scale + scale_range * 2
        overflow_mask = scales.amax(dim=-1)>overflow_thr
        if overflow_mask.sum().item() > 0:
            scales_overflow = scales[overflow_mask]
            overflow_term = torch.mean((scales_overflow - overflow_thr) ** 2)
        else:
            overflow_term = .0
        
        underflow_thr = 1.0e-7
        underflow_mask = scales.amin(dim=-1)<underflow_thr
        if underflow_mask.sum().item() > 0:
            scales_underflow = scales[underflow_mask]
            underflow_term = torch.mean((scales_underflow - underflow_thr) ** 2)
        else:
            underflow_term = .0

        n_underflow = underflow_mask.sum().item()
        n_overflow = overflow_mask.sum().item()

        loss = 1.0e-4 * mse_term + overflow_term + underflow_term
        return loss

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.FloatTensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).cuda()
        return gauss / gauss.sum()

    def create_window(self, window_size, feat_dim):
        window_1d = self.gaussian(window_size, 1.5)[:,None]
        window_2d = torch.mm(window_1d, window_1d.permute(1,0))[None,None,:,:]
        window_2d = window_2d.repeat(feat_dim,1,1,1)
        return window_2d

    def forward(self, img_out, img_target, bbox=None, mask=None, window_size=11):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if mask is not None:
            img_out = img_out * mask
            img_target = img_target * mask
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]

        window = self.create_window(window_size, feat_dim)
        mu1 = F.conv2d(img_out, window, padding=window_size//2, groups=feat_dim)
        mu2 = F.conv2d(img_target, window, padding=window_size//2, groups=feat_dim)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img_out*img_out, window, padding=window_size//2, groups=feat_dim) - mu1_sq
        sigma2_sq = F.conv2d(img_target*img_target, window, padding=window_size//2, groups=feat_dim) - mu2_sq
        sigma1_sigma2 = F.conv2d(img_out*img_target, window, padding=window_size//2, groups=feat_dim) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1_sigma2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map

# image perceptual loss (LPIPS. https://github.com/richzhang/PerceptualSimilarity)
class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.lpips = lpips.LPIPS(net='vgg').cuda()

    def forward(self, img_out, img_target, bbox=None):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]
        img_out = img_out * 2 - 1 # [0,1] -> [-1,1]
        img_target = img_target * 2 - 1 # [0,1] -> [-1,1]
        loss = self.lpips(img_out, img_target)
        return loss

def get_interior_selected(src_face_normal, src_xyz, trg_xyz):
    '''
    :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    '''
    N1, N2 = src_xyz.size(1), trg_xyz.size(1)

    # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
    NN_src_xyz = src_xyz
    NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
    NN_src_normal = src_face_normal

    interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false
    return interior

class HOIInitLoss(nn.Module):
    def __init__(self, opt, obj_gaussians, hand_gaussians):
        super(HOIInitLoss, self).__init__()
        self.opt_with_coarse = opt.opt_with_coarse
        if self.opt_with_coarse:
            obj_triangles = obj_gaussians.coarse_faces.unsqueeze(0)
            self.obj_triangles = obj_triangles
        else:
            faces = obj_gaussians.faces
            if faces is not None:
                obj_triangles = obj_gaussians.faces.unsqueeze(0)
                self.obj_triangles = obj_triangles
            else:
                self.obj_triangles = None

        hand_triangles = hand_gaussians.mano_layer.faces.astype(int)
        hand_triangles = torch.from_numpy(hand_triangles).unsqueeze(0)
        self.hand_triangles = hand_triangles
        self.palm_idxs = [34, 62, 194, 238, 288, 386, 397, 604, 614, 625, 141, 496, 507, 114, 126, 755, 763, 350, 439, 573, 690]
        self.palm_idxs_wopinky = [34, 62, 194, 238, 288, 386, 397, 141, 496, 507, 114, 126, 755, 763, 350, 439, 573]
        self.back_idxs = [191, 144, 212, 283, 270, 388, 405, 289, 590, 628, 290, 498, 516, 229, 29, 708, 724, 311, 423, 534, 651]

    def forward(self, hoi_gaussians, obj_sample_idx, obj_sample_normal,
                pene_weight=1.0, contact_loss_weight=1.0, normal_loss_weight=0.1):
        xyz = hoi_gaussians.get_xyz
        n_obj_gs = hoi_gaussians.obj_gaussians._xyz.shape[0]
        obj_xyz, hand_xyz = xyz[:n_obj_gs], xyz[n_obj_gs:]

        if self.opt_with_coarse:
            coarse_xyz = hoi_gaussians.get_coarse_xyz
            n_coarse_obj_gs = hoi_gaussians.obj_gaussians.coarse_verts.shape[0]
            coarse_obj_xyz = coarse_xyz[:n_coarse_obj_gs]
        
        # prepare v, mesh, normal
        obj_v_tensor = obj_xyz.unsqueeze(0)
        hand_v_tensor = hand_xyz.unsqueeze(0)
        if torch.isnan(obj_v_tensor).any().item():
            print(f'obj_v_tensor has nan value')
        if torch.isnan(hand_v_tensor).any().item():
            print(f'hand_v_tensor has nan value')
        device = obj_v_tensor.device
        hand_mesh = Meshes(hand_v_tensor, self.hand_triangles.to(device))
        rhand_normal = hand_mesh.verts_normals_packed().view(1, -1, 3)

        # get penetration info
        nn_dists, nn_idx = get_NN(obj_v_tensor, hand_v_tensor)
        interior = get_interior(rhand_normal, hand_v_tensor, obj_v_tensor, nn_idx)
        if torch.isnan(nn_dists).any().item():
            print(f'nn_dists has nan value')

        # minimize general penetration
        nn_dist = nn_dists.sqrt()
        nn_dist_interior = nn_dist[interior] # [n_penetrate]

        if interior.sum() > 0:
            penet_loss = nn_dist_interior.mean() # mean-depth penalty
        else:
            penet_loss = torch.zeros(1).cuda().sum()
        penet_loss = penet_loss * pene_weight

        if self.opt_with_coarse:
            obj_mesh = Meshes(obj_v_tensor, self.obj_triangles.to(device))
            obj_normal = obj_mesh.verts_normals_packed().view(1, -1, 3)
        else:
            obj_mesh = Meshes(obj_v_tensor, self.obj_triangles.to(device))
            obj_normal = obj_mesh.verts_normals_packed().view(1, -1, 3)

        # hand contact loss (w.r.t. obj sample point)
        hand_contact_idxs = [194, 755, 60] # ff, th, ff-th
        hand_contact_verts = hand_xyz[None,hand_contact_idxs]
        hand_contact_normals = rhand_normal[0:1,hand_contact_idxs]
        
        obj_contact_vert = obj_xyz[None,obj_sample_idx:obj_sample_idx+1]
        obj_contact_normal = obj_sample_normal[None,:]

        palm_dist = F.mse_loss(hand_contact_verts, obj_contact_vert, reduction='none').mean()
        hand_contact_loss = palm_dist * contact_loss_weight
        
        # general hand contact loss
        if self.obj_triangles is None:
            contact_loss_avg = contact_loss = torch.zeros(1).cuda().sum()
        else:
            mano_joints = hand_v_tensor[0:1,self.palm_idxs]
            rhand_contact_joint = hoi_gaussians.rhand_contact_joint.detach()
            # set rhand_contact_joint to all 1
            rhand_contact_joint = torch.ones_like(rhand_contact_joint)
            rhand_dist_xyz, rhand_nn_dist, obj_pc_contact_rhand = get_hand2obj_dist(mano_joints, obj_v_tensor, obj_normal)
            contact_loss = F.mse_loss(mano_joints, obj_pc_contact_rhand, reduction='none').mean(dim=0)
        
            rhand_contact_joint_sum = rhand_contact_joint.sum()
            if rhand_contact_joint_sum.item() == 0:
                #print(f'rhand_nn_dist.shape: {rhand_nn_dist.shape}, rhand_contact_joint.shape: {rhand_contact_joint.shape}')
                nn_dists_sorted, dist_indices = torch.sort(rhand_nn_dist[0])
                close_thr = nn_dists_sorted[5]
                rhand_contact_joint[rhand_nn_dist[0]<close_thr] = True
                hoi_gaussians.rhand_contact_joint = rhand_contact_joint
                
            contact_loss_masked = contact_loss * rhand_contact_joint.unsqueeze(1).float()
            contact_loss_avg = 0.1 * contact_loss_weight * contact_loss_masked.sum() / rhand_contact_joint.sum()
            hand_contact_loss = hand_contact_loss + contact_loss_avg
        

        # penalize back contact
        """
        back_idxs = [212, 708, 145]
        #back_idxs = self.back_idxs
        back_joints = hand_xyz[None,back_idxs]
        back_dist = F.mse_loss(back_joints, obj_contact_vert, reduction='none').mean()
        back_reg_loss = torch.clamp(palm_dist - back_dist, min=0.0)
        """
        """
        rhand_dist_xyz, rhand_nn_dist, obj_pc_contact_rhand_back = get_hand2obj_dist(back_joints, obj_v_tensor, obj_normal)
        back_contact_loss = F.mse_loss(back_joints, obj_pc_contact_rhand_back, reduction='none').mean(dim=0)

        back_reg_max = 0.001
        back_reg_loss = torch.clamp(back_reg_max-back_contact_loss, .0, back_reg_max)
        back_reg_loss = contact_loss_weight * back_reg_loss.mean()
        """
        if self.obj_triangles is not None:
            #back_joints = mano_out.vertices[0:1,self.back_idxs] * hand_rescaling + hand_trans
            back_joints = hand_v_tensor[0:1,self.back_idxs]
            rhand_dist_xyz, rhand_nn_dist, obj_pc_contact_rhand_back = get_hand2obj_dist(back_joints, obj_v_tensor, obj_normal)
            back_contact_loss = F.mse_loss(back_joints, obj_pc_contact_rhand_back, reduction='none').mean(dim=0)
            back_contact_loss_masked = back_contact_loss * rhand_contact_joint.unsqueeze(1).float()
            back_reg_loss = torch.clamp(contact_loss_masked - back_contact_loss_masked, .0) * rhand_contact_joint.unsqueeze(1).float()
            back_reg_loss = contact_loss_weight * back_reg_loss.sum() / rhand_contact_joint.sum()
        else:
            back_reg_loss = .0

        # hand_contact_verts normals should face to the obj_contact_vert
        contact_center = torch.mean(hand_xyz[hand_contact_idxs], dim=0, keepdim=True)
        hand_obj_dir = F.normalize(obj_contact_vert.squeeze(1) - contact_center, dim=-1)
        palm_normals = rhand_normal[0, self.palm_idxs]
        mean_palm_normal = F.normalize(torch.mean(palm_normals, dim=0, keepdim=True), dim=-1)
        normal_loss = F.mse_loss(mean_palm_normal, hand_obj_dir, reduction='none').mean() * normal_loss_weight

        #total_loss = penet_loss + hand_contact_loss
        loss_dict = {'penetration': penet_loss,
                    'hand_contact': hand_contact_loss,
                    'back_loss': back_reg_loss,
                    'normal_loss': normal_loss,
                    }
        return loss_dict

class HandGraspLoss(nn.Module):
    def __init__(self, obj_gaussians, hand_gaussians, opt_with_coarse=False):
        super(HandGraspLoss, self).__init__()
        self.opt_with_coarse = opt_with_coarse
        if opt_with_coarse:
            obj_triangles = obj_gaussians.coarse_faces.unsqueeze(0)
            self.obj_triangles = obj_triangles
        else:
            faces = obj_gaussians.faces
            if faces is not None:
                obj_triangles = obj_gaussians.faces.unsqueeze(0)
                self.obj_triangles = obj_triangles
            else:
                self.obj_triangles = None

        hand_triangles = hand_gaussians.mano_layer.faces.astype(int)
        hand_triangles = torch.from_numpy(hand_triangles).unsqueeze(0)

        self.hand_triangles = hand_triangles
        self.palm_idxs = [34, 62, 194, 238, 288, 386, 397, 604, 614, 625, 141, 496, 507, 114, 126, 755, 763, 350, 439, 573, 690]
        self.back_idxs = [191, 144, 212, 283, 270, 388, 405, 289, 590, 628, 290, 498, 516, 229, 29, 708, 724, 311, 423, 534, 651]

    def forward(self, hoi_gaussians, pene_weight=1.0, contact_loss_weight=1.0, ik_loss_weight=1.0, normal_align_weight=0.1, return_dict=False):
        xyz = hoi_gaussians.get_xyz
        n_obj_gs = hoi_gaussians.obj_gaussians._xyz.shape[0]
        obj_xyz, hand_xyz = xyz[:n_obj_gs], xyz[n_obj_gs:]

        if self.opt_with_coarse:
            coarse_xyz = hoi_gaussians.get_coarse_xyz
            n_coarse_obj_gs = hoi_gaussians.obj_gaussians.coarse_verts.shape[0]
            coarse_obj_xyz = coarse_xyz[:n_coarse_obj_gs]

        # prepare v, mesh, normal
        obj_v_tensor = obj_xyz.unsqueeze(0)
        hand_v_tensor = hand_xyz.unsqueeze(0)
        device = obj_v_tensor.device
        hand_mesh = Meshes(hand_v_tensor, self.hand_triangles.to(device))
        rhand_normal = hand_mesh.verts_normals_packed().view(1, -1, 3)

        # get penetration info
        nn_dists, nn_idx = get_NN(obj_v_tensor, hand_v_tensor)
        interior = get_interior(rhand_normal, hand_v_tensor, obj_v_tensor, nn_idx)

        # minimized general penetration
        nn_dist = nn_dists.sqrt()
        nn_dist_interior = nn_dist[interior]

        if interior.sum() > 0:
            penet_loss = nn_dist_interior.mean()
        else:
            penet_loss = torch.zeros(1).to(device).sum()
        
        penet_loss = penet_loss * pene_weight

        # contact distance loss
        hand_rescaling = hoi_gaussians.hand_gaussians.hand_rescaling
        # Use MANO for contact
        mano_layer = hoi_gaussians.hand_gaussians.mano_layer
        root_pose = hoi_gaussians.hand_gaussians.root_pose
        if root_pose.shape[-1] == 6:
            root_pose = matrix_to_axis_angle(rotation_6d_to_matrix(root_pose))
        
        hand_pose = hoi_gaussians.hand_gaussians.hand_pose
        if hand_pose.shape[1] == 6:
            hand_pose = matrix_to_axis_angle(rotation_6d_to_matrix(hand_pose))
        hand_pose = hand_pose.reshape(1, -1)
        hand_shape = hoi_gaussians.hand_gaussians.shape_param
        hand_trans = hoi_gaussians.hand_gaussians.trans + hoi_gaussians.hand_gaussians.hand_trans
        if hand_trans.dim()==2:
            hand_trans = hand_trans.unsqueeze(1)
        mano_out = mano_layer(global_orient=root_pose, hand_pose=hand_pose, betas=hand_shape)
        mano_joints = mano_out.vertices[0:1,self.palm_idxs] * hand_rescaling + hand_trans
        
        if self.opt_with_coarse:
            obj_v_tensor = coarse_obj_xyz.unsqueeze(0)
        
        
        if self.obj_triangles is None:
            contact_loss_avg = torch.zeros(1).to(device).sum()
            ik_loss = torch.zeros(1).to(device).sum()
            back_reg_loss = torch.zeros(1).to(device).sum()
        else:
            if self.opt_with_coarse:
                """
                coarse_obj_mesh = Meshes(coarse_obj_xyz.unsqueeze(0), self.obj_triangles.to(device))
                coarse_obj_normal = coarse_obj_mesh.verts_normals_packed().view(1, -1, 3)
                # assign coarse normals to fine vertices by nearest neighbor
                _, nn_idx = get_NN(obj_v_tensor, coarse_obj_xyz.unsqueeze(0))
                obj_normal = batched_index_select(coarse_obj_normal, nn_idx)
                """
                obj_mesh = Meshes(obj_v_tensor, self.obj_triangles.to(device))
                obj_normal = obj_mesh.verts_normals_packed().view(1, -1, 3)
            else:
                obj_mesh = Meshes(obj_v_tensor, self.obj_triangles.to(device))
                obj_normal = obj_mesh.verts_normals_packed().view(1, -1, 3)
            #print(f'obj_normal shape: {obj_normal.shape}, obj_v_tensor shape: {obj_v_tensor.shape}')

            rhand_contact_joint = hoi_gaussians.rhand_contact_joint.detach()
            
            rhand_dist_xyz, rhand_nn_dist, obj_pc_contact_rhand = get_hand2obj_dist(mano_joints, obj_v_tensor, obj_normal)
            contact_loss = F.mse_loss(mano_joints, obj_pc_contact_rhand, reduction='none').mean(dim=0)
            
            contact_loss_masked = contact_loss * rhand_contact_joint.unsqueeze(1).float()
            if rhand_contact_joint.sum() > 0:
                contact_loss_avg = contact_loss_weight * contact_loss_masked.sum() / rhand_contact_joint.sum()
            else:
                contact_loss_avg = torch.zeros(1).to(device).sum()

            # penalize back contact
            back_joints = mano_out.vertices[0:1,self.back_idxs] * hand_rescaling + hand_trans
            rhand_dist_xyz, rhand_nn_dist, obj_pc_contact_rhand_back = get_hand2obj_dist(back_joints, obj_v_tensor, obj_normal)
            back_contact_loss = F.mse_loss(back_joints, obj_pc_contact_rhand_back, reduction='none').mean(dim=0)

            back_reg_max = 0.001
            back_reg_loss = torch.clamp(back_reg_max-back_contact_loss, .0, back_reg_max)
            back_reg_loss = contact_loss_weight * back_reg_loss.sum() / rhand_contact_joint.sum()
            
            # Normal Alignment Loss
            # 1. Get NN for both Normal Alignment and IK
            nn_dists, nn_idx = get_NN(mano_joints, obj_v_tensor)
            
            # 2. Gather object normals at contact points
            # obj_normal: [1, N_obj, 3] -> [1, 21, 3]
            obj_contact_normals = batched_index_select(obj_normal, nn_idx)
            
            # 3. Get hand normals at contact points
            # rhand_normal: [1, N_hand, 3] -> [1, 21, 3]
            hand_contact_normals = rhand_normal[:, self.palm_idxs, :]

            # 4. Compute Alignment Loss
            # We want hand normal to be opposite to object normal.
            # cos_sim should be -1. Minimize (1 + cos_sim)
            cos_sim = F.cosine_similarity(hand_contact_normals, obj_contact_normals, dim=-1) # [1, 21]
            normal_align_loss_term = (1.0 + cos_sim)
            
            # Mask by contact
            normal_align_loss_masked = normal_align_loss_term * rhand_contact_joint.unsqueeze(0).float()
            
            if rhand_contact_joint.sum() > 0:
                normal_align_loss = normal_align_loss_masked.sum() / rhand_contact_joint.sum()
            else:
                normal_align_loss = torch.zeros(1).to(device).sum()
            
            normal_align_loss = normal_align_loss * normal_align_weight

            # IK loss
            ik_target_joints = obj_pc_contact_rhand.detach().clone()
            # nn_dists, nn_idx = get_NN(mano_joints, obj_v_tensor) # Already computed
            interior = get_interior(obj_normal, obj_v_tensor, mano_joints, nn_idx)
            ik_mask = (rhand_contact_joint.unsqueeze(0) + interior) > 0
            
            if ik_mask.sum().item() > 0:
                ik_loss = torch.sum(torch.sum((mano_joints - ik_target_joints) ** 2, dim=-1) * ik_mask) / ik_mask.sum()
            else:
                ik_loss = torch.zeros(1).to(device).sum()
            ik_loss = ik_loss * ik_loss_weight
        
        loss = penet_loss + contact_loss_avg + ik_loss + back_reg_loss + normal_align_loss

        if return_dict:
            return {'penetration': penet_loss, 'hand_contact': contact_loss_avg, 'ik': ik_loss, 'back_reg': back_reg_loss, 'normal_align': normal_align_loss}
        else:
            return loss

class RotationLoss(nn.Module):
    """Physical stability cost for rotation refinement.
    
    Inspired by Zhao et al., "Stability-Driven Contact Reconstruction From
    Monocular Color Images", CVPR 2022.
    
    Computes the resultant force and torque residuals assuming static
    equilibrium: gravity acting on the object should be balanced by contact
    normal forces from the hand.
    """
    def __init__(self, obj_gaussians, hand_gaussians):
        super(RotationLoss, self).__init__()
        hand_triangles = hand_gaussians.mano_layer.faces.astype(int)
        self.hand_triangles = torch.from_numpy(hand_triangles).unsqueeze(0)

        faces = obj_gaussians.faces
        if faces is not None:
            self.obj_triangles = faces.unsqueeze(0)
        else:
            self.obj_triangles = None

    def forward(self, hoi_gaussians):
        xyz = hoi_gaussians.get_xyz
        n_obj_gs = hoi_gaussians.obj_gaussians._xyz.shape[0]
        obj_xyz, hand_xyz = xyz[:n_obj_gs], xyz[n_obj_gs:]
        device = obj_xyz.device

        obj_v = obj_xyz.unsqueeze(0)   # [1, N_obj, 3]
        hand_v = hand_xyz.unsqueeze(0) # [1, N_hand, 3]

        # Build hand mesh and compute normals
        hand_mesh = Meshes(hand_v, self.hand_triangles.to(device))
        hand_normal = hand_mesh.verts_normals_packed().view(1, -1, 3)

        # --- Identify contact points (obj verts close to / penetrating hand) ---
        nn_dists, nn_idx = get_NN(obj_v, hand_v)  # nn from obj -> hand
        interior = get_interior(hand_normal, hand_v, obj_v, nn_idx)

        nn_dist = nn_dists.sqrt()  # [1, N_obj]
        contact_threshold = 5.0e-3  # 5 mm
        close_mask = (nn_dist < contact_threshold).squeeze(0)  # [N_obj]
        contact_mask = (interior.squeeze(0) | close_mask)       # [N_obj]

        n_contact = contact_mask.sum().item()

        if n_contact == 0:
            # No contact: penalise with min-distance between obj and hand
            # (the farther apart, the more unstable)
            min_dist = nn_dist.min()
            loss = min_dist + 1.0  # baseline penalty for no contact
            return loss

        # --- Contact normals (hand surface normals at the NN hand verts) ---
        # nn_idx: [1, N_obj]  -> index into hand_v
        contact_nn_idx = nn_idx[0, contact_mask]  # [n_contact]
        contact_normals = hand_normal[0, contact_nn_idx]  # [n_contact, 3]
        # Ensure normals point outward from hand toward object
        contact_obj_pts = obj_xyz[contact_mask]  # [n_contact, 3]
        contact_hand_pts = hand_xyz[contact_nn_idx]  # [n_contact, 3]
        direction = contact_obj_pts - contact_hand_pts
        flip = (direction * contact_normals).sum(dim=-1) < 0  # where normal faces wrong way
        contact_normals[flip] = -contact_normals[flip]

        # --- Object center of mass ---
        obj_com = obj_xyz.mean(dim=0)  # [3]

        # --- Gravity ---
        # Unit gravity, magnitude = 1 (we only care about direction balance)
        g = torch.tensor([0.0, -1.0, 0.0], device=device)

        # --- Solve for normal force magnitudes ---
        # In static equilibrium: sum(f_i * n_i) + m*g = 0
        # We distribute force equally among contact normals as an approximation,
        # then measure how well the equilibrium is satisfied.
        # Assume unit object mass m=1.
        # Optimal per-contact force: f_i = -g / N  (projected onto the normal)
        # The actual supportable force along each normal: f_i = max(0, <-g, n_i>) / N
        # where the max(0, ...) ensures contact can only push (compression, not tension).

        n_c = float(n_contact)
        neg_g = -g  # [3], direction to counteract gravity
        # Per-contact force magnitude (clamped to non-negative = push only)
        force_mag = torch.clamp((contact_normals * neg_g.unsqueeze(0)).sum(dim=-1), min=0.0)  # [n_contact]
        force_mag_sum = force_mag.sum()

        if force_mag_sum < 1e-8:
            # Normals don't support gravity at all
            loss = torch.tensor(2.0, device=device)
            return loss

        # Normalise so total force magnitude equals gravity magnitude (1.0)
        force_mag_normalised = force_mag / force_mag_sum  # [n_contact]

        # Actual force vectors at each contact point
        force_vectors = contact_normals * force_mag_normalised.unsqueeze(1)  # [n_contact, 3]

        # --- Resultant force residual ---
        resultant_force = force_vectors.sum(dim=0) + g  # should be ~[0,0,0]
        force_residual = torch.norm(resultant_force)

        # --- Torque residual about CoM ---
        lever_arms = contact_obj_pts - obj_com.unsqueeze(0)  # [n_contact, 3]
        torques = torch.cross(lever_arms, force_vectors, dim=-1)  # [n_contact, 3]
        resultant_torque = torques.sum(dim=0)  # [3]
        torque_residual = torch.norm(resultant_torque)

        # --- Penetration penalty (mild) ---
        interior_mask = interior.squeeze(0)
        if interior_mask.sum() > 0:
            penetration_cost = nn_dist[0, interior_mask].mean()
        else:
            penetration_cost = 0.0

        loss = force_residual + torque_residual + 0.1 * penetration_cost
        return loss