"""
chunkwise t3bench script.
python tools/run_t3_chunkwise.py --gpu 0 --chunk 1
...
python tools/run_t3_chunkwise.py --gpu 0 --chunk 10
"""
import os
import glob
import argparse
from datetime import datetime
import sys
import gc
import torch
import copy

def run_cmd(cmd):
    os.system(cmd)

def run(args):
    obj_config_name = './configs/obj/it7000_obj.yaml'
    hand_config_name = './configs/hand/it7000_hand.yaml'
    hoi_config_name = './configs/hoi/it1000_hoi.yaml'
    obj_end_iter = 7000
    hoi_end_iter = 1000

    chunk = args.chunk

    initprompt_path = f'data/t3_prompt/initprompt_single.txt'
    with open(initprompt_path, 'r') as f:
        obj_init_lines = f.readlines()

    obj_prompt_path = f'data/t3_prompt/prompt_single.txt'
    with open(obj_prompt_path, 'r') as f:
        obj_lines = f.readlines()
    
    hand_prompt_path = f'data/t3_prompt/prompt_hand.txt'
    with open(hand_prompt_path, 'r') as f:
        hand_lines = f.readlines()

    hoi_prompt_path = f'data/t3_prompt/prompt_{args.group}_{chunk}-10.txt'
    with open(hoi_prompt_path, 'r') as f:
        hoi_lines = f.readlines()
    
    t2hoi_prompt_path = f'data/t3_prompt/t2hoi_prompt.txt'
    with open(t2hoi_prompt_path, 'r') as f:
        t2hoi_lines = f.readlines()

    base_output_dir = f'outputs_t3/{args.method}_{args.group}'
    os.makedirs(base_output_dir, exist_ok=True)
    
    gpu = args.gpu

    n_lines = len(hoi_lines)
    chunk_size = n_lines
    start_i = chunk_size * (chunk-1)
    end_i = chunk_size * chunk
    
    obj_init_lines = obj_init_lines[start_i:end_i]
    obj_lines = obj_lines[start_i:end_i]
    hand_lines = hand_lines[start_i:end_i]
    t2hoi_lines = t2hoi_lines[start_i:end_i]

    n_runs = 0
    start_time = datetime.now()
    for pi, hoi_prompt in enumerate(hoi_lines):
        print(f'[{pi}/{n_lines}] total progress')
        hoi_prompt = hoi_prompt.strip()
        obj_initprompt = obj_init_lines[pi].strip()
        obj_prompt = obj_lines[pi].strip()
        hand_prompt = hand_lines[pi].strip()
        t2hoi_prompt = t2hoi_lines[pi].strip()
        output_dir = os.path.join(base_output_dir, hoi_prompt.replace(' ', '_'))
        output_dir = output_dir.replace(',', '')
        output_dir = output_dir.replace("'s", "")
        output_dir = output_dir.replace("’s", "")

        # process quote
        t2hoi_prompt = t2hoi_prompt.replace("'s", "") # remove quote. it's hydra's fault
        t2hoi_prompt = t2hoi_prompt.replace("’s", "")

        # run obj
        obj_subdir = 'obj'
        obj_ckpt_path = os.path.join(output_dir, f'{obj_subdir}/chkpnt{obj_end_iter}.pth')
        if not os.path.exists(obj_ckpt_path):
            obj_cmd = f"CUDA_VISIBLE_DEVICES={gpu} python tools/run_obj.py " + \
                    f"--opt '{obj_config_name}' " + \
                    f"--prompt \"{obj_prompt}\" --initprompt \"{obj_initprompt}\" --output_dir \"{output_dir}\"/{obj_subdir}"
            print(obj_cmd)
            result = os.system(obj_cmd)
            if (result != 0):
                print('something bad happened when running obj fitting. terminate the script.')
                sys.exit()

        # run text2hoi
        t2hoi_pkl_path = os.path.join(output_dir, 't2hoi_results', 'text2hoi_res_min.pkl')
        if not os.path.exists(t2hoi_pkl_path):
        #if True:
            t2hoi_cmd = f"CUDA_VISIBLE_DEVICES={gpu} python tools/run_t2hoi.py dataset=grab " + \
                    f'+test_text=["{t2hoi_prompt}"] +nsamples=10 ' + \
                    f"+input_obj_path=[\"{output_dir}/{obj_subdir}/point_cloud/iteration_{obj_end_iter}/point_cloud.ply\"] " + \
                    f"hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled " + \
                    f"result_dir=\"{output_dir}\" "
            print(t2hoi_cmd)
            result = os.system(t2hoi_cmd)
            if (result != 0):
                print('something bad happened when running text2hoi inference. terminate the script.')
                sys.exit()
        
        # run hand
        hand_subdir = 'hand'
        hand_ckpt_path = os.path.join(output_dir, f'{hand_subdir}/chkpnt{obj_end_iter}.pth')
        hand_ply_path = os.path.join(output_dir, f'{hand_subdir}/point_cloud/iteration_{obj_end_iter}/point_cloud.ply')
        if not os.path.exists(hand_ckpt_path):
            hand_cmd = f"CUDA_VISIBLE_DEVICES={gpu} python tools/run_hand.py " + \
                    f"--opt '{hand_config_name}' " + \
                    f"--prompt \"{hand_prompt}\" --initprompt \"{hand_prompt}\" --output_dir \"{output_dir}\"/{hand_subdir} " + \
                    f"--text2hoi_pkl \"{t2hoi_pkl_path}\" "
            print(hand_cmd)
            result = os.system(hand_cmd)
            if (result != 0):
                print('something bad happened when running hand fitting. terminate the script.')
                sys.exit()
        
        # gc collect
        gc.collect()
        torch.cuda.empty_cache()

        # run vlrefine
        t2hoi_pkl_path = os.path.join(output_dir, 't2hoi_results', 'text2hoi_res_min.pkl')
        #print(f'config_name: {config_name}, t2hoi_pkl_path: {t2hoi_pkl_path}')
        if not os.path.exists(t2hoi_pkl_path):
            vl_cmd = f"CUDA_VISIBLE_DEVICES={gpu} python tools/run_vlrefine.py " + \
                    f"--opt '{hoi_config_name}' " + \
                    f"--obj_prompt \"{obj_prompt}\" " + \
                    f"--hand_prompt \"{hand_prompt}\" " + \
                    f"--hoi_prompt \"{hoi_prompt}\" " + \
                    f"--text2hoi_pkl \"{t2hoi_pkl_path}\" " + \
                    f"--obj_pth \"{output_dir}/{obj_subdir}/chkpnt{obj_end_iter}.pth\" " + \
                    f"--obj_ply \"{output_dir}/{obj_subdir}/point_cloud/iteration_{obj_end_iter}/point_cloud.ply\" " + \
                    f"--hand_pth \"{hand_ckpt_path}\" " + \
                    f"--output_dir \"{output_dir}/internvl_t\" "
            print(vl_cmd)
            result = os.system(vl_cmd)
            if (result != 0):
                print('something bad happened when running text2hoi inference. terminate the script.')
                sys.exit()
        
        # gc collect
        gc.collect()
        torch.cuda.empty_cache()

        # run hoi
        hoi_subdir = 'hoi'
        hoi_ckpt_path = os.path.join(output_dir, f'{hoi_subdir}/chkpnt{hoi_end_iter}.pth')
        if not os.path.exists(hoi_ckpt_path):
            hoi_cmd = f"CUDA_VISIBLE_DEVICES={gpu} python tools/run_hoi.py " + \
                    f"--opt '{hoi_config_name}' " + \
                    f"--obj_prompt \"{obj_prompt}\" " + \
                    f"--hand_prompt \"{hand_prompt}\" " + \
                    f"--hoi_prompt \"{hoi_prompt}\" " + \
                    f'--t2hoi_prompt \"{t2hoi_prompt}\" ' + \
                    f"--text2hoi_pkl \"{t2hoi_pkl_path}\" " + \
                    f"--obj_pth \"{output_dir}/{obj_subdir}/chkpnt{obj_end_iter}.pth\" " + \
                    f"--obj_ply \"{output_dir}/{obj_subdir}/point_cloud/iteration_{obj_end_iter}/point_cloud.ply\" " + \
                    f"--hand_pth \"{hand_ckpt_path}\" " + \
                    f"--output_dir \"{output_dir}/{hoi_subdir}\" --gpu_id {gpu}"
                    #f"--hand_ply \"{hand_ply_path}\" " + \
            
            print(hoi_cmd)
            result = os.system(hoi_cmd)
            if (result != 0):
                print('something bad happened when running hoi fitting. terminate the script.')
                sys.exit()
        
        n_runs += 1

        # gc collect
        gc.collect()
        torch.cuda.empty_cache()

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    total_seconds = int(elapsed_time.total_seconds()/n_runs)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    print(f'Elapsed time: {hours} hour {minutes} minute {seconds} seconds')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='hoi')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', type=str, default='thom')
    parser.add_argument('--chunk', type=int, default=1)
    args = parser.parse_args()
    
    run(args)
