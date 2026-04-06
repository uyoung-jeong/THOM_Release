import torch
import pickle
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
import numpy as np
import os
import trimesh

from scene.hand_gaussian_model import MANOParamDict
from .mano import mano

def read_mano_pkl(path):
    with open(path, 'rb') as f:
        t2hoi_data = pickle.load(f)
    for k,v in t2hoi_data.items():
        if isinstance(v, list):
            t2hoi_data[k] = np.array(v)
        elif isinstance(v, float):
            continue
    return t2hoi_data

# data is obtained from text2hoi inference
def get_mano_param(path):
    device = 'cuda'
    if path != '':
        try:
            #with open(path, 'rb') as f:
                #data = pickle.load(f)
            data = read_mano_pkl(path)

            pose_aa = data['hand_pose']
            trans = data['hand_trans']
            pose = matrix_to_rotation_6d(axis_angle_to_matrix(torch.tensor(pose_aa, dtype=torch.float32, device=device)))

            mano_params = {
                #'root_pose': torch.tensor(pose[:6], dtype=torch.float32, device=device).reshape(1,6),
                #'hand_pose': torch.tensor(pose[6:], dtype=torch.float32, device=device).reshape(15,6),
                'root_pose': pose[0:1],
                'hand_pose': pose[1:],
                #'trans': torch.tensor(trans, dtype=torch.float32, device=device).reshape(1,3),
                'trans': torch.zeros((1,3), device=device, dtype=torch.float32),
                }
        except FileNotFoundError as e:
            mano_params = {'root_pose': torch.zeros((1,6), device=device, dtype=torch.float32),
                        'hand_pose': torch.zeros((15,6), device=device, dtype=torch.float32),
                        'trans': torch.zeros((1,3), device=device, dtype=torch.float32),} 
    else:
        mano_params = {'root_pose': torch.zeros((1,6), device=device, dtype=torch.float32),
                        'hand_pose': torch.zeros((15,6), device=device, dtype=torch.float32),
                        'trans': torch.zeros((1,3), device=device, dtype=torch.float32),}
    mano_param_dict = MANOParamDict()
    with torch.no_grad():
        mano_param_dict.init(mano_params)
    return mano_param_dict

