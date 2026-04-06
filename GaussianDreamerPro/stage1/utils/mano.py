import sys
import numpy as np
import torch
from torch.nn import functional as F
import os.path as osp
import pickle
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import math
import os
import os.path as osp

import warnings
warnings.filterwarnings("ignore")

from utils.smplx import smplx

class MANO(object):
    def __init__(self, human_model_path='', subdivide_num=5):
        self.shape_param_dim = 100
        #self.expr_param_dim = 50
        self.layer_arg = {'create_global_orient': False, 
                        'create_body_pose': False, 'create_hand_pose': False,
                        'create_betas': False, 'create_transl': False,
                        'flat_hand_mean': True}
        if human_model_path == '':
            cur_dir = osp.dirname(os.path.abspath(__file__))
            root_dir = osp.join(cur_dir, '..', '..', '..')
            human_model_path = osp.join(root_dir, 'load', 'human_model_files')
        #self.layer = {gender: smplx.create(human_model_path, 'smplx', gender=gender, num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim, use_pca=False, use_face_contour=True, **self.layer_arg) for gender in ['neutral', 'male', 'female']}
        self.layer = {is_rhand: smplx.create(human_model_path, 'mano',
            is_rhand=is_rhand, num_betas=self.shape_param_dim,
            use_pca=False, **self.layer_arg) for is_rhand in [False,True]}

        #self.layer = {gender: self.get_expr_from_flame(self.layer[gender]) for gender in ['neutral', 'male', 'female']}
        self.vertex_num = 778
        self.face_orig = self.layer[1].faces.astype(np.int64)
        self.face = self.face_orig
        
        # MANO joint set
        self.joint_num = 16
        self.joints_name = ('wrist', 'forefinger1', 'forefinger2', 'forefinger3', 
                            'middle_finger1', 'middle_finger2', 'middle_finger3',
                            'pinky_finger1', 'pinky_finger2', 'pinky_finger3',
                            'ring_finger1', 'ring_finger2', 'ring_finger3',
                            'thumb1', 'thumb2', 'thumb3',
                            #'thumb4', 'forefinger4', 'middle_finger4', 'ring_finger4', 'pinky_finger4'
                            )
        self.joint_part = \
        {'hand': range(self.joints_name.index('wrist'), self.joints_name.index('thumb3')+1)}
        self.root_joint_idx = self.joints_name.index('wrist')
        self.neutral_hand_pose = torch.zeros((len(self.joint_part['hand'])-1,3)) # template pose in axis-angle representation (body pose without root joint)
        
        # subdivider
        self.subdivider_list = self.get_subdivider(subdivide_num)
        self.face_upsampled = self.subdivider_list[-1]._subdivided_faces.cpu().numpy()
        self.vertex_num_upsampled = int(np.max(self.face_upsampled)+1)
        #print(f"original face shape: {self.face.shape}, original vertex num: {self.vertex_num}")
        #print(f"{subdivide_num} upsampled face shape: {self.face_upsampled.shape}, upsampled vertex num: {self.vertex_num_upsampled}")
        """
        original face shape: [1538, 3], original vertex num: 778
        2 upsampled face shape: (24608, 3), upsampled vertex num: 12337
        3 upsampled face shape: (98432, 3), upsampled vertex num: 49281
        5 upsampled face shape: (1574912, 3), upsampled vertex num: 787713
        """

    def assign_new_subdivide_num(self, subdivide_num):
        self.subdivider_list = self.get_subdivider(subdivide_num)
        self.face_upsampled = self.subdivider_list[-1]._subdivided_faces.cpu().numpy()
        self.vertex_num_upsampled = int(np.max(self.face_upsampled)+1)
        print(f"{subdivide_num} upsampled face shape: {self.face_upsampled.shape}, upsampled vertex num: {self.vertex_num_upsampled}")

    def set_id_info(self, shape_param):
        self.shape_param = shape_param

    def get_subdivider(self, subdivide_num):
        vert = self.layer[1].v_template.float().cuda()
        face = torch.LongTensor(self.face).cuda()
        mesh = Meshes(vert[None,:,:], face[None,:,:])

        subdivider_list = [SubdivideMeshes(mesh)]
        for i in range(subdivide_num-1):
            mesh = subdivider_list[-1](mesh)
            subdivider_list.append(SubdivideMeshes(mesh))
        return subdivider_list

    def upsample_mesh(self, vert, feat_list=None):
        face = torch.LongTensor(self.face).cuda()
        mesh = Meshes(vert[None,:,:], face[None,:,:])
        if feat_list is None:
            for subdivider in self.subdivider_list:
                mesh = subdivider(mesh)
            vert = mesh.verts_list()[0]
            return vert
        else:
            feat_dims = [x.shape[1] for x in feat_list]
            feats = torch.cat(feat_list,1)
            for subdivider in self.subdivider_list:
                mesh, feats = subdivider(mesh, feats)
            vert = mesh.verts_list()[0]
            feats = feats[0]
            feat_list = torch.split(feats, feat_dims, dim=1)
            return vert, *feat_list

mano = MANO()
