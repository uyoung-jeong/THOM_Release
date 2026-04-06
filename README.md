# THOM: Generating Physically Plausible Hand-Object Meshes From Text
### [Project Page](https://uyoung-jeong.github.io/THOM_Project/) | [arxiv Paper](https://arxiv.org/abs/2604.02736)

[THOM: Generating Physically Plausible Hand-Object Meshes From Text](https://arxiv.org/abs/2604.02736) (CVPR 2026 Findings)

[Uyoung Jeong](https://uyoung-jeong.github.io/)<sup>1</sup>,
[Yihalem Yimolal Tiruneh](https://www.linkedin.com/in/yihalem-yimolal-tiruneh-852aab198/)<sup>1</sup>,
[Hyung Jin Chang](https://hyungjinchang.wordpress.com/)<sup>2</sup>,
[Seungryul Baek](https://sites.google.com/site/bsrvision00/)<sup>1</sup>,
[Kwang In Kim](https://sites.google.com/view/kimki)<sup>3</sup>

<sup>1</sup>UNIST &emsp;<sup>2</sup>University of Birmingham &emsp; <sup>3</sup>POSTECH

![block](./assets/fig1.png)

The generation of 3D hand-object interactions (HOIs) from text is crucial for dexterous robotic grasping and VR/AR content generation, requiring both high visual fidelity and physical plausibility. Nevertheless, the ill-posed problem of mesh extraction from text-generated Gaussians, and physics-based optimization on the erroneous meshes pose challenges. To address these issues, we introduce THOM, a training-free framework that generates photorealistic, physically plausible 3D HOI meshes without the need for a template object mesh. THOM employs a two-stage pipeline, initially generating the hand and object Gaussians, followed by physics-based HOI optimization. Our new mesh extraction method and vertex-to-Gaussian mapping explicitly assign Gaussian elements to mesh vertices, allowing topology-aware regularization. Furthermore, we improve the physical plausibility of interactions by VLM-guided translation refinement and contact-aware optimization. Comprehensive experiments demonstrate that THOM consistently surpasses state-of-the-art methods in terms of text alignment, visual realism, and interaction plausibility.

## :fire: Updates
- 2026-04-06: Initial code release.

## :hammer: Installation

This code has been tested on Ubuntu 20.04, cuda 11.8, python 3.11, and PyTorch 2.7 environment.
The code is designed to run with 24GB VRAM (e.g. NVIDIA RTX 3090).

### Environment setup
Set up the Gaussian Splatting environment:
```
git clone --recursive https://github.com/uyoung-jeong/THOM_Release.git
cd THOM_Release

conda create -n thom python==3.11.0 -y
conda activate thom
./setup_gs.sh

pip install ./GaussianDreamerPro/submodules/diff-gaussian-rasterization --no-build-isolation
pip install ./GaussianDreamerPro/submodules/diff-gaussian-rasterization_2dgs --no-build-isolation
pip install ./GaussianDreamerPro/submodules/simple-knn --no-build-isolation

./setup_extra.sh
```

If you encounter an error when compiling diff-gaussian-rasterization, try running `sudo apt-get install libglm-dev` [reference](https://github.com/graphdeco-inria/gaussian-splatting/issues/645).
Please check [GaussianDreamerPro github](https://github.com/hustvl/GaussianDreamerPro), and [Text2HOI github](https://github.com/JunukCha/Text2HOI) for detailed troubleshooting.

### Download
Download 'MANO_LEFT.pkl' and 'MANO_RIGHT.pkl' files from [MANO](https://mano.is.tue.mpg.de/) and place them under `./load/human_model_files/mano`.

Download [finetuned Shap-E](https://huggingface.co/datasets/tiange/Cap3D/blob/main/misc/our_finetuned_models/shapE_finetuned_with_330kdata.pth) by Cap3D, and place it in `./load`

Download [Text2HOI checkpoints]() pth files and place them in `./checkpoint/grab/`.

### Directory structure
```
configs
├─ obj
│  └─ it7000_obj.yaml
├─ hand
│  └─ it7000_hand.yaml
└─ hoi
   └─ it7000_hoi.yam

checkpoints
└─ grab
   ├─ contact_estimator.pth
   ├─ ...
   └─ texthom.pth

GaussianDreamerPro
├─ stage1
│  ├─ arguments
│  ├─ gaussian_renderer
│  ├─ guidance
│  │  └─ sd_utils.py
│  ├─ scene
│  │  ├─ dataset_readers.py
│  │  ├─ ...
│  │  └─ hoi_gaussian_model.py
│  ├─ utils
└─ submodules

load
├─ shapE_finetuned_with_330kdata.pth
├─ human_model_files
│  ├─ mano
│  │  ├─ MANO_LEFT.pkl
│  │  └─ MANO_RIGHT.pkl
...

tools
├─ run_obj.py (object generation)
...
└─ run_hoi.py (hoi generation)
```

## :rocket: HOI Generation Process
### Quick start
This script runs the full pipeline:
```
./scripts/run_hoi.sh "{object_prompt}" "{hand_prompt}" "{hoi_prompt}" "{text2hoi_prompt}" {gpu_id} {output_directory}
```

Example command:
```
./scripts/run_hoi.sh "A hamburger" "A right hand" "A right hand eating a hamburger" "Eat a hamburger with the right hand." 0 output/eating_hamburger
```

### Step-by-step process
```
# Object generation
python tools/run_obj.py --opt ./configs/obj/it7000_obj.yaml --prompt 'A hamburger' --initprompt 'A hamburger' --output_dir output/eating_hamburger/obj

# Pose initialization
python tools/run_t2hoi.py dataset=grab '+test_text=[Eat a hamburger with the right hand.]' +nsamples=10 '+input_obj_path=['\''output/eating_hamburger/obj/meshify_5000_coarse.ply'\'']' hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled result_dir=output/eating_hamburger

# Hand generation
python tools/run_hand.py --opt ./configs/hand/it7000_hand.yaml --prompt 'A right hand' --initprompt 'A right hand' --output_dir output/eating_hamburger/hand --text2hoi_pkl output/eating_hamburger/t2hoi_results/text2hoi_res_min.pkl

# HOI refinement
python tools/run_vlrefine.py --opt configs/hoi/it1000_hoi.yaml --obj_prompt 'A hamburger' --hand_prompt 'A right hand' --hoi_prompt 'A right hand eating a hamburger' --text2hoi_pkl output/eating_hamburger/t2hoi_results/text2hoi_res_min.pkl --obj_pth output/eating_hamburger/obj/chkpnt7000.pth --obj_ply output/eating_hamburger/obj/point_cloud/iteration_7000/point_cloud.ply --hand_pth output/eating_hamburger/hand/chkpnt7000.pth --output_dir output/eating_hamburger/internvl_t

# HOI generation
python tools/run_hoi.py --opt ./configs/hoi/it1000_hoi.yaml --obj_prompt 'A hamburger' --hand_prompt 'A right hand' --hoi_prompt 'A right hand eating a hamburger' --t2hoi_prompt 'Eat a hamburger with the right hand.' --text2hoi_pkl output/eating_hamburger/t2hoi_results/text2hoi_res_min_vlm_t.pkl --obj_pth output/eating_hamburger/obj/chkpnt7000.pth --obj_ply output/eating_hamburger/obj/point_cloud/iteration_7000/point_cloud.ply --hand_pth output/eating_hamburger/hand/chkpnt7000.pth --output_dir output/eating_hamburger/hoi
```

## :bulb: Citation
```
coming soon
```

## :pray: Acknowledgements
Our repository is built upon the following works:
- [GaussianDreamerPro](https://github.com/hustvl/GaussianDreamerPro)
- [Text2HOI](https://github.com/JunukCha/Text2HOI)
- [ExAvatar](https://github.com/mks0601/ExAvatar_RELEASE)
