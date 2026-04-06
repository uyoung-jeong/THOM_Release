"""
Infer initial HOI.
This file assumes that you have a coarse mesh file of GaussianDreamerPro object Gaussians.
"""
import os
import os.path as osp

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import hydra
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import time

import torch
import torch.nn.functional as F

import pymeshlab as ml
import trimesh
import pickle
from tqdm import tqdm

import sys
text2hoi_dir = osp.join(osp.dirname(osp.abspath(osp.dirname(__file__))), 'Text2HOI')
#print(f'text2hoi_dir: {text2hoi_dir}. {osp.exists(text2hoi_dir)}')
sys.path.append(text2hoi_dir)

from lib.models.mano import build_mano_aa
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import (
    get_object_hand_info_thom,
    get_valid_mask_bunch, 
    proc_results, 
)
from lib.utils.model_utils import (
    build_refiner, 
    build_model_and_diffusion, 
    build_seq_cvae, 
    build_mpnet, 
    build_pointnetfeat, 
    build_contact_estimator, 
)
#from lib.models.object import build_object_model
from lib.models.object_thom import build_object_thom_model
from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.utils.file import (
    make_save_folder, 
    save_mesh_obj, 
)
from lib.utils.proc import (
    proc_obj_feat_final, 
    proc_cond_contact_estimator, 
    proc_refiner_input, 
    farthest_point_sample,
    proc_numpy
)
from lib.utils.rot import (
    rot6d_to_rotmat,
    rotation_matrix_to_angle_axis
)

from lib.utils.proc_output import (
    get_pytorch3d_meshes,
    get_hand_joints_w_tip,
    get_transformed_obj_pc,
    get_NN, get_interior
)
from lib.utils.loss import get_penet_hand_obj_loss

# https://github.com/JunukCha/Text2HOI/blob/main/preprocessing_grab_object.py
def load_gdp_object(texts, obj_paths, invert_y=False, rescale_factor=0.1354):
    assert len(obj_paths)==1
    n_target_vertices = 4000
    new_mesh_paths = []
    for mesh_path in obj_paths:
        # sample 4000 vertices
        mesh_dir = osp.dirname(mesh_path)
        ms = ml.MeshSet(verbose=0)
        ms.load_new_mesh(mesh_path)
        m = ms.current_mesh()
        TARGET = n_target_vertices
        numFaces = 100 + 2 * TARGET
        while (ms.current_mesh().vertex_number() > TARGET):
            ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
            numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)
        m = ms.current_mesh()

        #print('sampled output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')

        verts_raw = m.vertex_matrix()

        if invert_y:
            verts_raw[:,1] = -verts_raw[:,1] # invert y axis

        verts = torch.tensor(verts_raw, dtype=torch.float32, device='cuda').unsqueeze(0)

        # rescale vertices
        #verts = verts - verts.amin()
        #verts = (verts - 0.5) * 0.1354
        #verts = verts * 0.1354
        
        maxmin = verts.amax(dim=1)[0] - verts.amin(dim=1)[0]
        max_axis = torch.argmax(maxmin).item()
        scale_raw = maxmin[max_axis]
        verts = verts / (maxmin[max_axis])
        verts = verts * rescale_factor
        #verts = verts * 0.0677

        #print(f'rescaled vertex min: {verts.min()}, vmax: {verts.max()}')
        scale_new = verts.amax(dim=1)[0,max_axis].item() - verts.amin(dim=1)[0,max_axis].item()

        normal = torch.tensor(m.vertex_normal_matrix(), dtype=torch.float32, device='cuda').unsqueeze(0)
        normal = normal / torch.norm(normal, dim=2, keepdim=True)
        
        new_mesh_fname = osp.basename(mesh_path).replace('.ply', '_simplify.ply')
        new_mesh_path = osp.join(mesh_dir, new_mesh_fname)

        new_mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy()[0], faces=m.face_matrix(), process=False)
        ply_data = trimesh.exchange.ply.export_ply(new_mesh, vertex_normal=True)
        with open(new_mesh_path, 'wb') as f:
            f.write(ply_data)
        #ms.save_current_mesh(new_mesh_path)
        print(f"simplified mesh saved at {new_mesh_path}")

        point_set = farthest_point_sample(verts, 1024)
        sampled_pc = verts[0, point_set[0]].cpu().numpy()
        sampled_normal = normal[0, point_set[0]].cpu().numpy()

        point_set_np = point_set[0].cpu().numpy()
        #verts_raw_sample = verts_raw[point_set_np]

        obj_dict = {"obj_pcs": sampled_pc,
                    "obj_pc_normals": sampled_normal, 
                    "point_sets": point_set_np, 
                    "obj_path": new_mesh_path, 
                    "verts_raw": verts_raw,
                    "scale_factor": scale_new/scale_raw
                }
        obj_model = build_object_thom_model(obj_dict)
        
    return obj_model


@hydra.main(version_base=None, config_path="../Text2HOI/configs", config_name="config")
@torch.no_grad()
def main(config):
    start_time = time.time()
    #print(OmegaConf.to_yaml(config))
    OmegaConf.to_yaml(config)
    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name

    invert_y = True

    save_root = os.path.join(config.result_dir, 't2hoi_results')
    result_folder = make_save_folder(save_root)
    pkl_save_path = osp.join(result_folder, f'text2hoi_res_min.pkl')
    print(f"pickle file path: {pkl_save_path}")
    
    #save_obj = config.save_obj
    nsamples = config.nsamples # 4
    fps = config.fps # 30
    max_nframes = data_config.max_nframes # 150
    text = config.test_text
    hand_nfeats = config.texthom.hand_nfeats # 99
    obj_nfeats = config.texthom.obj_nfeats # 9

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).cuda()

    # vertex idxs for the palm side of the MANO joints
    palm_idxs = np.array([34, 62, 194, 238, 288, 386, 397, 604, 614, 625, 141, 496, 507, 114, 126, 755, 763, 350, 439, 573, 690])
    back_idxs = np.array([191, 144, 212, 283, 270, 388, 405, 289, 590, 628, 290, 498, 516, 229, 29, 708, 724, 311, 423, 534, 651])

    refiner = build_refiner(config)
    texthom, diffusion \
        = build_model_and_diffusion(config, lhand_layer, rhand_layer, test=True)
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()
    mpnet = build_mpnet(config)
    seq_cvae = build_seq_cvae(config, test=True)
    pointnet = build_pointnetfeat(config, test=True)
    #pointnet.eval()
    contact_estimator = build_contact_estimator(config, test=True)
    rescale_factor = config.rescale_factor
    #object_model = build_object_model(data_config.data_obj_pc_path)
    object_model = load_gdp_object(text, config.input_obj_path, invert_y, rescale_factor)
    
    renderer = Renderer(device="cuda", camera=f"{dataset_name}_front")

    is_lhand, is_rhand, \
    obj_pc_org, obj_pc_normal_org, \
    normalized_obj_pc, point_sets, \
    obj_cent, obj_scale, \
    obj_verts, obj_faces, \
    obj_top_idx, obj_pc_top_idx, verts_raw \
        = get_object_hand_info_thom(
            object_model, 
            clip_model, 
            text, 
            data_config.obj_root, 
            data_config,
            mpnet,  
        )
    # is_lhand, is_rhand: [1]
    # normalized_obj_pc.shape: [1, 1024, 3]
    # point_sets.shape: [1, 1024]
    # obj_cent.shape: [1, 3]
    # obj_scale.shape: [1]
    # obj_verts.shape: [[2492,3]]
    # obj_faces.shape: [[4495, 3]]
    
    bs, npts = normalized_obj_pc.shape[:2] # 1, 1024
    
    enc_text = encoded_text(clip_model, text) # [1, 512]

    obj_feat = pointnet(normalized_obj_pc) # [1, 1024, 1088]
    
    min_criterion = 99999.9
    batch_num = len(text)//64 + 1 # 1
    pbar = tqdm(range(nsamples))
    for sample_idx in pbar:
        for batch_idx in range(batch_num):
            ecn_text_batch = enc_text[batch_idx*64:(batch_idx+1)*64] # [1, 512]
            is_lhand_batch = is_lhand[batch_idx*64:(batch_idx+1)*64] # [1]
            is_rhand_batch = is_rhand[batch_idx*64:(batch_idx+1)*64] # [1]
            obj_cent_batch = obj_cent[batch_idx*64:(batch_idx+1)*64] # [1,3]
            obj_scale_batch = obj_scale[batch_idx*64:(batch_idx+1)*64] # [1]
            obj_feat_batch = obj_feat[batch_idx*64:(batch_idx+1)*64] # [1, 1024, 1088]
            enc_text_batch = enc_text[batch_idx*64:(batch_idx+1)*64] # [1, 512]
            obj_pc_org_batch = obj_pc_org[batch_idx*64:(batch_idx+1)*64] # [1, 1024, 3]
            obj_pc_normal_org_batch = obj_pc_normal_org[batch_idx*64:(batch_idx+1)*64] # [1, 1024, 3]
            normalized_obj_pc_batch = normalized_obj_pc[batch_idx*64:(batch_idx+1)*64] # [1, 1024, 3]
            point_sets_batch = point_sets[batch_idx*64:(batch_idx+1)*64] # [1, 1024]
            obj_verts_batch = obj_verts[batch_idx*64:(batch_idx+1)*64] # [[2492, 3]]
            obj_faces_batch = obj_faces[batch_idx*64:(batch_idx+1)*64] # [[4995, 3]]
            verts_raw_batch = verts_raw[batch_idx*64:(batch_idx+1)*64]
            if dataset_name == "arctic":
                obj_top_idx_batch = obj_top_idx[batch_idx*64:(batch_idx+1)*64]
                obj_pc_top_idx_batch = obj_pc_top_idx[batch_idx*64:(batch_idx+1)*64]
            else:
                obj_top_idx_batch = None
                obj_pc_top_idx_batch = None
                
            duration = seq_cvae.decode(ecn_text_batch) # shape:[1,1]. [[0~1]]
            duration *= config.max_duration
            duration = torch.clamp(duration.long(), 1, 150)

            valid_mask_lhand, valid_mask_rhand, valid_mask_obj \
                = get_valid_mask_bunch(
                    is_lhand_batch, is_rhand_batch, 
                    max_nframes, duration
                )
            # valid_mask_lhand.shape: [1, 150]
            # valid_mask_rhand.shape: [1, 150]
            # valid_mask_obj.shape: [1, 150]
            obj_feat_final, est_contact_map = proc_obj_feat_final(
                contact_estimator,  
                obj_scale_batch, obj_cent_batch, 
                obj_feat_batch, enc_text_batch, npts, 
                config.texthom.use_obj_scale_centroid, # True
                config.contact.use_scale, # True
                config.texthom.use_contact_feat, # True
            )
            # obj_feat_final.shape: [1, 2052]
            # est_contact_map.shape: [1, 1024]
            coarse_x_lhand, coarse_x_rhand, coarse_x_obj \
                = diffusion.sampling(
                    texthom, obj_feat_final, 
                    enc_text_batch, max_nframes, 
                    hand_nfeats, obj_nfeats, 
                    valid_mask_lhand, 
                    valid_mask_rhand, 
                    valid_mask_obj, 
                    device=torch.device("cuda")
                )
            # coarse_x_lhand.shape: [1, 150, 90]
            # coarse_x_rhand.shape: [1, 150, 99]
            # coarse_x_obj.shape: [1, 150, 9]
            
            if est_contact_map is None:
                condition = proc_cond_contact_estimator(
                    obj_scale_batch, obj_feat_batch, enc_text_batch, 
                    npts, config.contact.use_scale
                )
                est_contact_map = contact_estimator.decode(condition)
                est_contact_map = (est_contact_map[..., 0] > 0.5).long()
            
            input_lhand, input_rhand, refined_x_obj, obj_pc_contact_lhand_psuedo, obj_pc_contact_rhand_psuedo, \
            lhand_contact_joint_mask, rhand_contact_joint_mask \
                = proc_refiner_input(
                    coarse_x_lhand, coarse_x_rhand, coarse_x_obj, 
                    lhand_layer, rhand_layer, obj_pc_org_batch, obj_pc_normal_org_batch, 
                    valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
                    est_contact_map, dataset_name, return_psuedo_gt=True, obj_pc_top_idx=obj_pc_top_idx_batch
                )
            
            # input_lhand.shape: [1, 150, 2273]
            # input_rhand.shape: [1, 150, 2273]
            # refined_x_obj.shape: [1, 150, 9]
            
            refined_x_lhand, refined_x_rhand \
                = refiner(
                    input_lhand, input_rhand,  
                    valid_mask_lhand=valid_mask_lhand, 
                    valid_mask_rhand=valid_mask_rhand, 
                )
            # refined_x_lhand.shape: [1, 150, 99]
            # refined_x_rhand.shap: [1, 150, 99]
            for text_idx in range(normalized_obj_pc_batch.shape[0]):
                #print(f"Batch #{batch_idx} Samples #{sample_idx} Text #{text_idx}, duration: {duration[text_idx].item()}")
                pbar.set_description(f"Batch #{batch_idx} Samples #{sample_idx} Text #{text_idx}")
                
                is_lhand_text = is_lhand_batch[text_idx]
                is_rhand_text = is_rhand_batch[text_idx]
                obj_verts_text = obj_verts_batch[text_idx] # [n_v_obj, 3]
                obj_faces_text = obj_faces_batch[text_idx] # [n_f_obj, 3]
                verts_raw_text = verts_raw_batch[text_idx]
                if dataset_name == "arctic":
                    obj_top_idx_text = obj_top_idx_batch[text_idx]
                else:
                    obj_top_idx_text = None
                
                text_duration = duration[text_idx].item()
                
                refined_x_lhand_sampled = refined_x_lhand[text_idx][:text_duration] # [n_duration, 99]
                refined_x_rhand_sampled = refined_x_rhand[text_idx][:text_duration] # [n_duration, 99]
                refined_x_obj_sampled = refined_x_obj[text_idx][:text_duration] # [n_duration, 9]
                
                refined_obj_verts_tf, refined_lhand_verts, lhand_faces, \
                refined_rhand_verts, rhand_faces = \
                    proc_results(
                        refined_x_lhand_sampled, refined_x_rhand_sampled, refined_x_obj_sampled, 
                        obj_verts_text, lhand_layer, rhand_layer, 
                        is_lhand_text, is_rhand_text, 
                        dataset_name, obj_top_idx_text
                    )
                # refined_obj_verts_tf.shape: [n_duration, 2492, 3]
                # refined_lhand_verts, lhand_faces: None or [n_duration, 778, 3], [1538, 3]
                # refined_rhand_verts.shape: [n_duration, 778, 3]
                # rhand_faces.shape: [1538, 3]
                
                if is_lhand_text:
                    refined_lhand_verts[:, :, :2] = refined_lhand_verts[:, :, :2] - refined_obj_verts_tf[0, :, :2].mean(0)[None, None]
                
                if is_rhand_text:
                    refined_rhand_verts[:, :, :2] = refined_rhand_verts[:, :, :2] - refined_obj_verts_tf[0, :, :2].mean(0)[None, None]
                
                refined_obj_verts_tf[:, :, :2] = refined_obj_verts_tf[:, :, :2] - refined_obj_verts_tf[0, :, :2].mean(0)[None, None]
                
                sampling_method = 'min_dist' # ['torch_cost', 'min_dist']
                #sampling_method = 'torch_cost'

                if sampling_method == 'min_dist':
                    # get a frame with the most contact
                    dist = (refined_obj_verts_tf[:,:,None,:] - refined_rhand_verts[:,None,:,:])**2
                    dist = dist.sum(dim=3).amin(dim=1)
                    n_close = torch.sum(dist<1.0e-4, dim=1)

                    if n_close.amax().item()>0:
                        sample_fid = n_close.argmax().item()
                    else:
                        sample_fid = dist.mean(dim=1).argmin().item()
                else:
                    # sample a frame with the lowest cost
                    # penetration loss
                    #print(f'refined_x_rhand_sampled.shape: {refined_x_rhand_sampled.shape}, refined_obj_verts_tf.shape: {refined_obj_verts_tf.shape}')
                    rhand_mesh, rhand_verts = get_pytorch3d_meshes(refined_x_rhand_sampled.unsqueeze(0), rhand_layer)
                    rhand_normal = rhand_mesh.verts_normals_packed().view(-1, 778, 3)
                    _, npts_trans = refined_obj_verts_tf.shape[:2]
                    #transf_obj_pc = get_transformed_obj_pc(refined_x_obj_sampled.unsqueeze(0), refined_obj_verts_tf, dataset_name, obj_top_idx_text)
                    transf_obj_pc = refined_obj_verts_tf
                    transf_obj_pc = transf_obj_pc.reshape(-1, npts_trans, 3)
                    #print(f'rhand_verts.shape: {rhand_verts.shape}, transf_obj_pc.shape: {transf_obj_pc.shape}')
                    nn_dist, nn_idx = get_NN(transf_obj_pc, rhand_verts)
                    interior = get_interior(rhand_normal, rhand_verts, transf_obj_pc, nn_idx)
                    nn_dist = nn_dist.sqrt()
                    #print(f'nn_dist.shape: {nn_dist.shape}, interior.shape: {interior.shape}')
                    penet_losses = []
                    for fi in range(len(nn_dist)):
                        if interior[fi].sum() > 0:
                            penet_losses.append(nn_dist[fi][interior[fi]].mean())
                        else:
                            penet_losses.append(torch.zeros(1).cuda().mean())
                    penet_loss = torch.stack(penet_losses) * 10.0
                    #print(f'penet_loss.shape: {penet_loss.shape}') # [nframe]
                    #print(f'penet_loss[:10]: {penet_loss[:10]}')

                    # number of contacts
                    contacts_rhand = obj_pc_contact_rhand_psuedo[text_idx][:text_duration]
                    n_contacts = (contacts_rhand.sum(dim=2)>0).sum(dim=1).float()
                    #print(f'contacts_rhand.shape: {contacts_rhand.shape}') # [nframe, 21, 3]

                    # contact loss
                    contact_loss_type = 'l2'
                    rhand_joints = get_hand_joints_w_tip(refined_x_rhand_sampled.unsqueeze(0), rhand_layer)
                    transf_obj_pc = transf_obj_pc.permute(1,0,2).unsqueeze(2)
                    #print(f'rhand_joints.shape: {rhand_joints.shape}, transf_obj_pc.shape: {transf_obj_pc.shape}')
                    contact_loss = F.mse_loss(rhand_joints, transf_obj_pc, reduction='none')
                    contact_mask = (contacts_rhand.sum(2)>0).float()
                    filtered_contact_loss = contact_loss.mean(dim=3).mean(dim=0) * contact_mask
                    filtered_contact_loss = filtered_contact_loss.mean(dim=1)
                    #filtered_contact_loss = get_filtered_joint_loss_valid_mask(contact_loss, contact_mask, None)
                    #print(f'filtered_contact_loss.shape: {filtered_contact_loss.shape}')
                    #print(f'filtered_contact_loss[:10]: {filtered_contact_loss[:10]}')
                    
                    # object rotation
                    obj_rot6ds = refined_x_obj_sampled[:,3:9]
                    obj_rotmats = rot6d_to_rotmat(obj_rot6ds).reshape(-1, 3,3)
                    rot_cost = torch.clamp(0.2-obj_rotmats[:,0,0],0,None) + torch.clamp(0.2-obj_rotmats[:,2,2],0,None)
                    #print(f'obj_rotmats.shape: {obj_rotmats.shape}, rot_cost.shape: {rot_cost.shape}')
                    #print(f'rot_cost[:10]: {rot_cost[:10]}')

                    costs = penet_loss + rot_cost + filtered_contact_loss + 0.5 * (21.-n_contacts)/21.
                    sample_fid = torch.argmin(costs).item()
                    #print(f'sample_fid: {sample_fid}')
                    criterion = costs[sample_fid]

                # get hoi params for initialization
                rhand_params = refined_x_rhand_sampled[sample_fid]
                rhand_trans = rhand_params[:3]
                rhand_pose = rhand_params[3:].reshape(-1)

                obj_trans = refined_x_obj_sampled[sample_fid][:3].detach().cpu().numpy()
                obj_rot6d = refined_x_obj_sampled[sample_fid][3:9]
                obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(3,3).detach().cpu().numpy()

                rhand_pose_6d = rhand_pose.detach().cpu()
                rhand_pose_rotmat = rot6d_to_rotmat(rhand_pose_6d)

                rhand_pose_aa = rotation_matrix_to_angle_axis(rhand_pose_rotmat)

                if sampling_method == 'min_dist':
                    obj_mesh = trimesh.Trimesh(vertices=proc_numpy(refined_obj_verts_tf[0]),
                                                faces=proc_numpy(obj_faces_text)
                                            )
                    rhand_mesh = trimesh.Trimesh(vertices=proc_numpy(refined_rhand_verts[0]),
                                                faces=proc_numpy(rhand_faces)
                                            )

                    hand_manager = trimesh.collision.CollisionManager()
                    hand_manager.add_object('hand', rhand_mesh)

                    obj_manager = trimesh.collision.CollisionManager()
                    obj_manager.add_object('obj', obj_mesh)
                    hoi_min_dist = hand_manager.min_distance_other(obj_manager)
                    #print(f'hoi_min_dist: {hoi_min_dist}')
                    rot_cost = np.clip(0.2-obj_rotmat[0,0],0,None) + np.clip(0.2-obj_rotmat[2,2],0,None) # prefer not to flip the object

                    obj_verts_np = obj_mesh.vertices
                    rhand_verts_np = rhand_mesh.vertices
                    palm_vs = rhand_verts_np[palm_idxs[14:16]]
                    back_vs = rhand_verts_np[back_idxs[14:16]]
                    palm_back_mean = (palm_vs + back_vs) / 2.0
                    (obj_nn_points, nn_dists, obj_tri_id) = obj_mesh.nearest.on_surface(palm_back_mean)
                    palm_dist = np.linalg.norm(palm_vs - obj_nn_points, axis=1)
                    back_dist = np.linalg.norm(back_vs - obj_nn_points, axis=1)
                    palm_back_diff = palm_dist - back_dist

                    #palm_obj_nn, palm_dist, palm_tri_ids = trimesh.proximity.closest_point(obj_mesh, palm_vs)
                    #back_obj_nn, back_dist, back_tri_ids = trimesh.proximity.closest_point(obj_mesh, back_vs)
                    palm_back_cost = np.clip(palm_back_diff, 0, None).sum() * 300.0

                    criterion = hoi_min_dist * 10.0 + rot_cost * 2.0 + 10.0 + palm_back_cost
                    if hoi_min_dist == 0.0:
                        is_col, contacts = hand_manager.in_collision_other(obj_manager, return_data=True)
                        depths = np.array([e.depth for e in contacts])
                        max_depth = np.amax(depths)
                        depths_nonzero = depths[depths>0]
                        mean_depth = np.mean(depths_nonzero)
                        sum_depth = 1 / (1 + np.exp(-np.sum(depths_nonzero)))
                        n_contact = len(contacts)

                        # normals
                        if n_contact > 0:
                            contact_pts = np.stack([e.point for e in contacts])
                            
                            obj_cont_dist = np.sum((contact_pts[:,None,:] - obj_verts_np[None,:,:]) ** 2, axis=2)
                            obj_nn_idxs = np.argmin(obj_cont_dist, axis=1)
                            
                            rhand_cont_dist = np.sum((contact_pts[:,None,:] - rhand_verts_np[None,:,:]) ** 2, axis=2)
                            rhand_nn_idxs = np.argmin(rhand_cont_dist, axis=1)

                            obj_normal_np = obj_mesh.vertex_normals
                            rhand_normal_np = rhand_mesh.vertex_normals

                            obj_normal_cont = obj_normal_np[obj_nn_idxs]
                            rhand_normal_cont = rhand_normal_np[rhand_nn_idxs]
                            obj_normal_cont_norm = np.linalg.norm(obj_normal_cont, axis=1)
                            rhand_normal_cont_norm = np.linalg.norm(rhand_normal_cont, axis=1)
                            dot_prod = np.sum(obj_normal_cont * rhand_normal_cont, axis=1)
                            cos_sim = dot_prod / (obj_normal_cont_norm * rhand_normal_cont_norm)
                            cos_sim_select = cos_sim[cos_sim>-0.7]
                            normal_cost = np.mean(cos_sim_select+0.7)

                            #print(f'max_depth: {max_depth}, mean_depth: {mean_depth}, normal_cost: {normal_cost}')
                        
                        #if n_contact > 20:
                        
                        if n_contact > 1:
                            criterion = mean_depth + normal_cost + rot_cost + sum_depth + palm_back_cost
                        else:
                            criterion = mean_depth + normal_cost + rot_cost + sum_depth + palm_back_cost + 5.0
                    print(f'criterion: {criterion}, hoi_min_dist: {hoi_min_dist*10.0}, rot_cost: {rot_cost*2.0}, palm_back_cost: {palm_back_cost}')
                # save for all the generation samples
                obj_pc_contact_rhand = obj_pc_contact_rhand_psuedo[text_idx][:text_duration][sample_fid]
                rhand_contact_joint = rhand_contact_joint_mask[text_idx][:text_duration][sample_fid]

                save_dict = {
                    'hand_pose': rhand_pose_aa.detach().cpu().numpy().tolist(), # [16, 3]
                    'hand_trans': rhand_trans.detach().cpu().numpy().tolist(), # [3]
                    'obj_rotmat': obj_rotmat.tolist(), # [3,3]
                    'obj_trans': obj_trans.tolist(), # [3]
                    'obj_rescaling': object_model.scale_factor.cpu().numpy().tolist(), # 0.0858
                    'obj_pc_contact_rhand': obj_pc_contact_rhand.detach().cpu().numpy().tolist(), # [21,3]
                    'rhand_contact_joint': rhand_contact_joint.detach().cpu().numpy().tolist(), # [21]
                    'criterion': criterion.tolist(),
                }
                with open(osp.join(result_folder, f'text2hoi_res_{sample_idx}.pkl'), 'wb') as f:
                    pickle.dump(save_dict, f)
                if criterion < min_criterion:
                    min_criterion = criterion

                    text_underbar = '_'.join(text[text_idx].split(' '))
                    #pkl_save_path = osp.join(result_folder, f'{text_underbar}_{sample_idx}.pkl')
                    #pkl_save_path = osp.join(result_folder, f'{text_underbar}_min.pkl')
                    with open(pkl_save_path, 'wb') as f:
                        pickle.dump(save_dict, f)

                    if invert_y:
                        refined_obj_verts_tf[:,:,1] = -refined_obj_verts_tf[:,:,1]

                    # save hoi mesh list
                    hoi_mesh_save_dir = osp.join(result_folder, 'obj_file')
                    os.makedirs(hoi_mesh_save_dir, exist_ok=True)
                    #hoi_mesh_save_path = osp.join(hoi_mesh_save_dir,  f"{text_underbar}_{sample_idx}_hoi.ply")
                    hoi_mesh_save_path = osp.join(hoi_mesh_save_dir,  f"{text_underbar}_min.ply")
                    
                    if sampling_method == 'min_dist':
                        hoi_mesh = trimesh.util.concatenate(obj_mesh, rhand_mesh)
                        hoi_ply_data = trimesh.exchange.ply.export_ply(hoi_mesh, vertex_normal=True)
                        with open(hoi_mesh_save_path, 'wb') as f:
                            f.write(hoi_ply_data)
                        
    elapsed_time = time.time() - start_time
    print(f'elapsed time: {elapsed_time}')


if __name__ == "__main__":
    main()