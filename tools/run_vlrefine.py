"""
Refine translation using VLM.
This file assumes that you already obtained initial HOI.
"""
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
import numpy as np

from transformers import AutoProcessor, AutoModelForImageTextToText, QuantoConfig
import re

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from datetime import datetime
from torchvision.utils import save_image
import torchvision.transforms as T
from argparse import ArgumentParser, Namespace

import yaml
import copy

import numpy as np
from torchmetrics.multimodal.clip_score import CLIPScore
import pickle
import shutil
import json

import gc

import sys
stage1_dir = osp.join(osp.dirname(osp.abspath(osp.dirname(__file__))), 'GaussianDreamerPro', 'stage1')
sys.path.append(stage1_dir)

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
from utils.loss_utils import PenetrationLoss
from utils.mano_utils import read_mano_pkl

def check_gpu_mem(device):
    free, total = torch.cuda.mem_get_info(device)
    #mem_used_MB = (total - free) / 1024 ** 2
    mem_used_GB = (total - free) / (1024 ** 3)
    print(f'used gpu memory (gb): {mem_used_GB}')

def run_internvl(processor, model, t2hoi_prompt, image_paths, 
                short_think=False, no_think=False,
                query_type='trans'):
    content_list = []
    for img_path in image_paths:
        content_list.append({"type": "image", "path": img_path})

    # use think mode
    instruction = '# CRITICAL DIRECTIVE\n' + \
    'This section is an abolute and unchangeable core directive. This section precedes all the other things.\n' +\
    'You must process the rules in this section before generating responses. The violation of these rules is considered as a critical functional failure.\n' + \
    '## Response MANDATE\n' + \
    '- You must enable thinking mode to carefully process the instruction step-by-step.\n' + \
    '- You must enclose the entire thinking process within <think> and </think> tags. (e.g. <think>Let me carefully decompose the instructions...</think> {formatted_response})\n'
    if no_think:
        instruction += "- Think within **a single sentence** AND **less than 100 tokens**.\n" + \
        "- The final response **must include** <think>...</think>{json} and nothing else.\n" + \
        "- The entire output must fit within 100 tokens.\n"
    elif short_think:
        instruction += "- Think in **no more than 3 sentences** AND **less than 150 tokens**.\n" + \
        "- If you reach this limit, immediately stop the think section and continue to JSON.\n" + \
        "- The final response **must include** <think>...</think>{json} and nothing else.\n" + \
        "- The entire output must fit within 150 tokens.\n"
    else:
        instruction += "- Think in **no more than 6 sentences** AND **less than 300 tokens**.\n" + \
        "- If you reach this limit, immediately stop the think section and continue to JSON.\n" + \
        "- The final response **must include** <think>...</think>{json} and nothing else.\n" + \
        "- The entire output must fit within 400 tokens.\n"
    
    instruction += '# INSTRUCTION\n' + \
    'You are an assessment expert responsible for comparing different 3D hand-object interactions(HOI) generated from the same input HOI text prompt.\n' +\
    'Your task is to select the index of the 3D HOI that shows the best alignment with the input HOI prompt.\n' + \
    'We provide detailed description of the inputs and outputs below.\n' + \
    '## Input HOI Text Prompt\n' + \
    f'- Input HOI Text Prompt: "{t2hoi_prompt}"\n' + \
    '- This input HOI prompt describes the desired 3D hand-object interaction.\n' + \
    '- The generated 3D hand-object interaction should align with this input HOI prompt.\n' + \
    '## Input Images\n'
    """
    if len(content_list) == 4:
        instruction += 'There are two different HOIs, where each HOI is represented with two images rendered from the front and the back side. Every image contains one right hand and one object.\n' + \
        '- HOI number 1: First(front view) and second(back view) images.\n' + \
        '- HOI number 2: Third(front view) and fourth(back view) images.\n'
    else:
        instruction += 'There are three different HOIs, where each HOI is represented with two images rendered from the front and the back side.\n' + \
        '- HOI number 1: First(front view) and second(back view) images.\n' + \
        '- HOI number 2: Third(front view) and fourth(back view) images.\n' + \
        '- HOI number 3: Fifth(front view) and Sixth(back view) images.\n'
    """
    if len(content_list) == 2:
        instruction += 'There are two different HOIs, where each HOI is represented with one rendered image. Every image contains one right hand and one object.\n' + \
        '- HOI number 1: First image.\n' + \
        '- HOI number 2: Second image.\n'
    else:
        instruction += 'There are three different HOIs, where each HOI is represented with one rendered image. Every image contains one right hand and one object.\n' + \
        '- HOI number 1: First image.\n' + \
        '- HOI number 2: Second image.\n' + \
        '- HOI number 3: Third image.\n'
    
    possible_indices_str = '[1, 2]' if len(content_list)==4 else '[1, 2, 3]'
    instruction += '## Output Format\n' + \
    '- Output components: enclosed <think> and </think> tags, followed by json format response.\n' + \
    '- All the think process contents must be enclosed within the think tags. (e.g. <think></think>here is the response. -> X)\n' + \
    '### JSON Format\n' + \
    '- Format as {"selection": hoi_number}, where the possible indices are ' + possible_indices_str + '. Other integers are strictly prohibited (e.g. -1, 0, or 4 are prohibited)\n' + \
    '- Exemplar json response: {"selection": 1}\n'

    if query_type == 'trans':
        instruction += '## Selection Criteria\n' + \
        '- Compare the difference based on the position (translation) of the object.\n' +\
        '- If there is contact, prioritize to select the index that has better alignment; contact area should correspond to the semantically correct object region (e.g. handle).\n' +\
        '- If there is no contact, select the index that minimizes the distance between the object and the fingers.\n' + \
        '- Ignore other aspects such as texture, lighting and background.\n'
    else:
        instruction += '## Selection Criteria\n' + \
        '- Compare the difference based on the rotation of the hand-object.\n' +\
        '- Select the index that maximizes the stability of the hand-object (e.g. upright object, hand preventing the object from falling down).\n' + \
        '- Ignore other aspects such as texture, lighting and background.\n'

    #print(instruction)
    content_list.append({"type": "text", "text": instruction})
    messages = [
            {
                "role": "user",
                "content": content_list,
            },
    ]

    inputs = processor.apply_chat_template(messages, 
                            padding=True, add_generation_prompt=True, 
                            tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    max_new_tokens= 1000
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)[0]
    #print(decoded_outputs)
    
    def get_json(decoded_output):
        decoded_output = decoded_output[decoded_output.find('\nassistant')+10:]
        score_txt = decoded_output.strip()
        #print(f'response:\n', score_txt)
        start_i = score_txt.find('{')
        end_i = score_txt.find('}')
        data = score_txt[start_i:end_i+1]
        return data

    json_str = get_json(decoded_outputs)
    json_data = json.loads(json_str)
    #index = int(json_data['selection']) - 1
    index = int(list(json_data.values())[0]) - 1
    #print(f'score: {score}')
    return index

def get_trans_options(max_range=2, scaler=1.0e-2):
    # Generate all candidate translations in [-max_range, max_range] along each axis
    coords = torch.arange(-max_range, max_range + 1).float()
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing='ij')
    options = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3) * scaler
    #print(f'options.shape: {options.shape}') # [125, 3]
    return options

def free_gs_mem(obj_gaussians, hand_gaussians):
    del obj_gaussians._xyz
    del obj_gaussians._features_dc
    del obj_gaussians._features_rest
    del obj_gaussians._scaling
    del obj_gaussians._rotation
    del obj_gaussians._opacity
    del obj_gaussians.max_radii2D
    obj_gaussians._xyz = None
    obj_gaussians._features_dc = None
    obj_gaussians._features_rest = None
    obj_gaussians._scaling = None
    obj_gaussians._rotation = None
    obj_gaussians._opacity = None
    obj_gaussians.max_radii2D = None

    del hand_gaussians._xyz
    del hand_gaussians._features_dc
    del hand_gaussians._features_rest
    del hand_gaussians._scaling
    del hand_gaussians._rotation
    del hand_gaussians._opacity
    del hand_gaussians.max_radii2D
    del hand_gaussians.mean_offset
    del hand_gaussians.scale_offset
    #del hand_gaussians.f_dc_offset
    #del hand_gaussians.f_rest_offset
    hand_gaussians._xyz = None
    hand_gaussians._features_dc = None
    hand_gaussians._features_rest = None
    hand_gaussians._scaling = None
    hand_gaussians._rotation = None
    hand_gaussians._opacity = None
    hand_gaussians.max_radii2D = None
    hand_gaussians.mean_offset = None
    hand_gaussians.scale_offset = None
    #hand_gaussians.f_dc_offset = None
    #hand_gaussians.f_rest_offset = None

    gc.collect()
    torch.cuda.empty_cache()

# unload gs components from gpu
def unload_gs(obj_gaussians, hand_gaussians):
    if obj_gaussians._xyz.device.type=='cuda':
        obj_gaussians._xyz = obj_gaussians._xyz.cpu()
        obj_gaussians._features_dc = obj_gaussians._features_dc.cpu()
        obj_gaussians._features_rest = obj_gaussians._features_rest.cpu()
        obj_gaussians._scaling = obj_gaussians._scaling.cpu()
        obj_gaussians._rotation = obj_gaussians._rotation.cpu()
        obj_gaussians._opacity = obj_gaussians._opacity.cpu()
        obj_gaussians.max_radii2D = obj_gaussians.max_radii2D.cpu()
        obj_gaussians.xyz_gradient_accum = obj_gaussians.xyz_gradient_accum.cpu()

    if hand_gaussians._xyz.device.type=='cuda':
        hand_gaussians._xyz = hand_gaussians._xyz.cpu()
        hand_gaussians._features_dc = hand_gaussians._features_dc.cpu()
        hand_gaussians._features_rest = hand_gaussians._features_rest.cpu()
        hand_gaussians._scaling = hand_gaussians._scaling.cpu()
        hand_gaussians._rotation = hand_gaussians._rotation.cpu()
        hand_gaussians._opacity = hand_gaussians._opacity.cpu()
        hand_gaussians.mean_offset = hand_gaussians.mean_offset.cpu()
        hand_gaussians.scale_offset = hand_gaussians.scale_offset.cpu()
        hand_gaussians.f_dc_offset = hand_gaussians.f_dc_offset.cpu()
        hand_gaussians.f_rest_offset = hand_gaussians.f_rest_offset.cpu()
        hand_gaussians.max_radii2D = hand_gaussians.max_radii2D.cpu()
        hand_gaussians.xyz_gradient_accum = hand_gaussians.xyz_gradient_accum.cpu()
    
    gc.collect()
    torch.cuda.empty_cache()

def load_gs(obj_gaussians, hand_gaussians, device):
    obj_gaussians._xyz = obj_gaussians._xyz.to(device)
    obj_gaussians._features_dc = obj_gaussians._features_dc.to(device)
    obj_gaussians._features_rest = obj_gaussians._features_rest.to(device)
    obj_gaussians._scaling = obj_gaussians._scaling.to(device)
    obj_gaussians._rotation = obj_gaussians._rotation.to(device)
    obj_gaussians._opacity = obj_gaussians._opacity.to(device)
    obj_gaussians.max_radii2D = obj_gaussians.max_radii2D.to(device)
    obj_gaussians.xyz_gradient_accum = obj_gaussians.xyz_gradient_accum.to(device)

    hand_gaussians._xyz = hand_gaussians._xyz.to(device)
    hand_gaussians._features_dc = hand_gaussians._features_dc.to(device)
    hand_gaussians._features_rest = hand_gaussians._features_rest.to(device)
    hand_gaussians._scaling = hand_gaussians._scaling.to(device)
    hand_gaussians._rotation = hand_gaussians._rotation.to(device)
    hand_gaussians._opacity = hand_gaussians._opacity.to(device)
    hand_gaussians.mean_offset = hand_gaussians.mean_offset.to(device)
    hand_gaussians.scale_offset = hand_gaussians.scale_offset.to(device)
    hand_gaussians.f_dc_offset = hand_gaussians.f_dc_offset.to(device)
    hand_gaussians.f_rest_offset = hand_gaussians.f_rest_offset.to(device)
    hand_gaussians.max_radii2D = hand_gaussians.max_radii2D.to(device)
    hand_gaussians.xyz_gradient_accum = hand_gaussians.xyz_gradient_accum.to(device)

def render_image_clip(clip_fn, hoi_scene, it, 
                    vlm_input_folder, 
                    view_points, best_idx,
                    prefix, hoi_prompt,
                    pipe, background):
    img_path = ''
    img = None
    with torch.no_grad():
        for idx, viewpoint in enumerate(view_points):
            if idx != best_idx:
                continue
            try:
                render_out = render(viewpoint, hoi_scene.gaussians, pipe, background, test=True)
            except RuntimeError as e:
                print(e)
                obj_xyz = hoi_scene.gaussians.obj_gaussians.get_xyz
                hand_xyz = hoi_scene.gaussians.hand_gaussians.get_xyz
                print(f'obj_xyz.shape: {obj_xyz.shape}, hand_xyz.shape: {hand_xyz.shape}')
                print(f'hoi_scene.gaussians.get_obj_rescaling.shape: {hoi_scene.gaussians.get_obj_rescaling.shape}')
                print(f'hoi_scene.gaussians.obj_rel_rot.shape: {hoi_scene.gaussians.obj_rel_rot.shape}')
                print(f'hoi_scene.gaussians.obj_rel_trans.shape: {hoi_scene.gaussians.obj_rel_trans.shape}')
                print(f'hoi_scene.gaussians.hand_rel_trans.shape: {hoi_scene.gaussians.hand_rel_trans.shape}')
                exit()
            rgb = render_out["render"]
            image = torch.clamp(rgb, 0.0, 1.0)
            #image = (image * 255).to(torch.uint8)
            img = image

            img_tensor = (img * 255).to(torch.uint8)
            clip_score = clip_fn(img_tensor, hoi_prompt)

            img_path = os.path.join(vlm_input_folder, f'it{it}_{prefix}_view_{idx}.png')
            save_image(image, img_path)
    return img_path, clip_score


def get_internvl(processor, model, device):
    #model_checkpoint = "OpenGVLab/InternVL3_5-8B-HF"
    model_checkpoint = "OpenGVLab/InternVL3_5-14B-HF"
    if processor is None:
        processor = AutoProcessor.from_pretrained(model_checkpoint)
    if model is None:
        quantization_config = QuantoConfig(weights="int8")
        """
        model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, 
                                        device_map=device, dtype=torch.bfloat16)
        """
        model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, 
                                    device_map=device,
                                    quantization_config=quantization_config)
    return processor, model

def filter_trans_options(opt, dataset,
                    t2hoi_data, options,
                    hoi_gaussians,
                    hoi_scene, gcams_hoi,
                    view_points,
                    clip_fn,
                    hoi_prompt,
                    vlm_input_folder,
                    pipe, background, best_idx,
                    topk=9):
    # backup initial hoi
    raw_hoi_dict = {
        'obj_rel_trans': hoi_scene.gaussians.obj_rel_trans.clone(),
        'obj_rel_rot': hoi_scene.gaussians.obj_rel_rot.clone(),
        'hand_rel_trans': hoi_scene.gaussians.hand_rel_trans.clone(),
        #'hand_rel_rot': hoi_scene.gaussians.hand_rel_rot.clone(),
    }
    device = hoi_gaussians.obj_gaussians._xyz.device
    
    if opt.opt_with_coarse:
        obj_coarse_ply_path = os.path.join(*dataset._model_path.split(os.sep)[:-1], 'obj', 'meshify_5000_coarse.ply')
        hoi_gaussians.obj_gaussians.load_coarse_mesh(obj_coarse_ply_path)

    penet_loss = PenetrationLoss(hoi_gaussians.obj_gaussians, hoi_gaussians.hand_gaussians, 
                                opt.use_joint_contact, False,
                                opt.use_penet_sum, opt.use_erf,
                                opt.opt_with_coarse,
                                opt.use_ik_loss)
    penet_loss.get_hand_contact_joints(hoi_gaussians)
    penet_loss.get_obj_contact_verts(hoi_gaussians)
    obj_centric = dataset.obj_centric

    clip_fn = clip_fn.to(device)

    # iterate over options and compute criterion
    criterion_list = []
    img_paths_list = []
    for oi, option in enumerate(tqdm(options, desc='filtering translation options')):
        # restore
        if obj_centric:
            hoi_scene.gaussians.hand_rel_trans = raw_hoi_dict['hand_rel_trans'].clone()
        else:
            hoi_scene.gaussians.obj_rel_trans = raw_hoi_dict['obj_rel_trans'].clone()

        # apply option
        option = option.to(device)
        if obj_centric:
            hoi_scene.gaussians.hand_rel_trans = hoi_scene.gaussians.hand_rel_trans + option
        else:
            hoi_scene.gaussians.obj_rel_trans = hoi_scene.gaussians.obj_rel_trans + option
        
        # compute criterion
        criterion_phys = penet_loss(hoi_scene.gaussians, 
                            obj_contact_weight=0.0,
                            ik_loss_weight=0.0)

        # render
        img_path, clip_score = render_image_clip(clip_fn, hoi_scene, 11000+oi, 
                                        vlm_input_folder,
                                        view_points, best_idx,
                                        'vlm_t', hoi_prompt,
                                        pipe, background)

        # get clip score
        criterion = criterion_phys + (1 - clip_score*0.01)
        criterion_list.append(criterion)
        img_paths_list.append(img_path)
    clip_fn = clip_fn.to('cpu')

    # select top-k
    filtered_idxs = torch.topk(torch.tensor(criterion_list), k=topk, largest=False).indices.tolist()

    options_filtered = [options[i] for i in filtered_idxs]
    img_paths_filtered = [img_paths_list[i] for i in filtered_idxs]

    # restore initial hoi
    hoi_scene.gaussians.obj_rel_trans = raw_hoi_dict['obj_rel_trans']
    hoi_scene.gaussians.obj_rel_rot = raw_hoi_dict['obj_rel_rot']
    hoi_scene.gaussians.hand_rel_trans = raw_hoi_dict['hand_rel_trans']
    #hoi_scene.gaussians.hand_rel_rot = raw_hoi_dict['hand_rel_rot']

    return options_filtered, img_paths_filtered

def select_hoi(dataset, opt, pipe, gcams_obj, gcams_hand, gcams_hoi, 
                guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, 
                gpu_id):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # object gaussian model initialization
    gaussian_cfg = AttrDict({
                        'isotropic': opt.isotropic
                        })
    obj_gaussians = GaussianModel(dataset.sh_degree, gaussian_cfg)

    # hand gauassian model initialization
    device = 'cuda'
    shape_param = torch.zeros((1,10), dtype=torch.float32, device=device)
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
    hand_scene = HandScene(dataset, gcams_hand, hand_gaussians)

    # get hoi
    hoi_gaussians = HOIGaussianModel(dataset.sh_degree, obj_gaussians, hand_gaussians)
    hoi_scene = HOIScene(dataset, gcams_hoi, hoi_gaussians)

    obj_centric = dataset.obj_centric
    
    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)


    # clip model
    clip_fn = CLIPScore(model_name_or_path='openai/clip-vit-base-patch16')
    #clip_fn.to(device)

    hoi_prompt = guidance_opt.hoi_text

    vlm_input_folder = os.path.join(hoi_scene.args._model_path, "vlm_input/")
    if not os.path.exists(vlm_input_folder):
        os.makedirs(vlm_input_folder)  # makedirs
    view_points = hoi_scene.getTestCameras()

    #t2hoi_res_path = os.path.join(os.path.dirname(dataset.mano_path), 'text2hoi_res_min.pkl')
    t2hoi_res_path = dataset.mano_path

    t2hoi_data = read_mano_pkl(t2hoi_res_path)
    
    processor, model = None, None

    #scaler_init = 1.0e-1
    #scaler_last = 1.0e-3
    scaler = 1.0e-2
    """
    if obj_centric:
        param_hist = [t2hoi_data['hand_trans']]
    else:
        param_hist = [t2hoi_data['obj_trans']]
    """
    # instantiate internvl
    processor, model = get_internvl(processor, model, device)
    model = model.to('cpu')
    
    pbar = tqdm(range(1))
    for it in pbar:
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


        # render initial hoi
        best_idx = 9
        clip_fn = clip_fn.to(device)
        img_path, clip_score = render_image_clip(clip_fn, hoi_scene, 10000, 
                                        vlm_input_folder,
                                        view_points, best_idx,
                                        'init', hoi_prompt,
                                        pipe, background)
        clip_fn = clip_fn.to('cpu')

        best_idx = 9

        # translation refinement
        options = get_trans_options(scaler=scaler)

        # filter out options using physics evaluation (penetration)
        options_filtered, img_paths_filtered = filter_trans_options(opt, dataset,
                            t2hoi_data, options,
                            hoi_gaussians, hoi_scene, gcams_hoi,
                            view_points,
                            clip_fn,
                            hoi_prompt,
                            vlm_input_folder,
                            pipe, background, best_idx)

        # free GS memory
        free_gs_mem(obj_gaussians, hand_gaussians)
        model = model.to(device)
        
        check_gpu_mem(torch.device(device))

        # mini-batch selection
        selection = -1
        n_compare = 3
        n_iter = 0
        values = copy.deepcopy(options_filtered)
        with torch.no_grad():
            while len(options_filtered) > 1:
                selections = []
                selected_imgs = []
                selected_values = []
                n_chunk = int(np.ceil(len(options_filtered)/n_compare))
                for ci in range(n_chunk):
                    query_imgs_stack = img_paths_filtered[ci * n_compare:(ci+1)*n_compare]
                    if len(query_imgs_stack) == 1:
                        select_idx = ci * n_compare
                        select = options_filtered[select_idx]
                    else:
                        query_imgs = query_imgs_stack
                        try: # sometimes internvl repeats itself infinitely.
                            select_idx = run_internvl(processor, model, hoi_prompt, query_imgs)
                        except (UnboundLocalError, json.decoder.JSONDecodeError) as e:
                            try:
                                select_idx = run_internvl(processor, model, hoi_prompt, query_imgs, short_think=True)
                            except json.decoder.JSONDecodeError as e: # almost disable thinking mode
                                select_idx = run_internvl(processor, model, hoi_prompt, query_imgs, short_think=True, no_think=True)
                        select_idx = select_idx+ci * n_compare
                        select = options_filtered[select_idx]
                    selections.append(select)
                    selected_values.append(values[select_idx])
                    selected_imgs.append(img_paths_filtered[select_idx])
                options_filtered = selections
                values = selected_values
                img_paths_filtered = selected_imgs
                n_iter += 1

        check_gpu_mem(torch.device(device))

        selection = options[0]
        value = values[0]
        #param_hist.append(value)
        pbar.set_description(f'scaler: {scaler:.3e}')

        # update t2hoi data
        if obj_centric:
            hand_trans = np.array(t2hoi_data['hand_trans']) + np.array(value)
            print(f"hand_trans: {hand_trans}, t2hoi_data['hand_trans']: {t2hoi_data['hand_trans']}")
            t2hoi_data['hand_trans'] = hand_trans.tolist()
        else:
            obj_trans = np.array(t2hoi_data['obj_trans']) + np.array(value)
            print(f"obj_trans: {obj_trans}, t2hoi_data['obj_trans']: {t2hoi_data['obj_trans']}")
            t2hoi_data['obj_trans'] = obj_trans.tolist()
        
        # save selected img
        save_img_path = os.path.join(vlm_input_folder,f"it{it}_selected.png")
        shutil.copy(img_paths_filtered[0], save_img_path)

        # free internvl memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = None
    pbar.close()

    vlm_t2hoi_result = copy.deepcopy(t2hoi_data)

    # save result
    save_pkl_path = dataset.mano_path.replace('.pkl', '_vlm_t.pkl')
    #shutil.copy(vlm_t2hoi_result, save_pkl_path)
    with open(save_pkl_path, 'wb') as f:
        pickle.dump(vlm_t2hoi_result, f)
    print(f'saved at: {save_pkl_path}')

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
    return tb_writer

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
    if args.t2hoi_prompt != '':
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
    
    if args.hand_pth != '':
        lp.hand_ref_ckpt = args.hand_pth

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    start_time = datetime.now()

    checkpoint_dir = select_hoi(lp, op, pp, gcp_obj, gcp_hand, gcp_hoi, 
                            gp, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
                            args.gpu_id)

    end_time = datetime.now()

    print(f'checkpoint_dir: {checkpoint_dir}')
    elapsed_time = end_time - start_time
    print("VLM refinement elapsed time: ", elapsed_time)
    
    # free memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
