import sys
sys.path.append('./gaussian_splatting')
import os
import torch
import plotly.graph_objs as go
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render as gs_render
from gaussian_splatting.scene.dataset_readers import fetchPly
from sugar_utils.spherical_harmonics import SH2RGB
from .cameras import CamerasWrapper, load_gs_cameras

from plyfile import PlyData
import numpy as np
from torch import nn
try:
    from utils.sh_utils import RGB2SH
except ModuleNotFoundError as e:
    from gaussian_splatting.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2

class ModelParams(): 
    """Parameters of the Gaussian Splatting model.
    Largely inspired by the original implementation of the 3D Gaussian Splatting paper:
    https://github.com/graphdeco-inria/gaussian-splatting
    """
    def __init__(self):
        self.sh_degree = 3
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
    
        
class PipelineParams():
    """Parameters of the Gaussian Splatting pipeline.
    Largely inspired by the original implementation of the 3D Gaussian Splatting paper:
    https://github.com/graphdeco-inria/gaussian-splatting
    """
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class OptimizationParams():
    """Parameters of the Gaussian Splatting optimization.
    Largely inspired by the original implementation of the 3D Gaussian Splatting paper:
    https://github.com/graphdeco-inria/gaussian-splatting
    """
    def __init__(self):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002


class GaussianSplattingWrapper:
    """Class to wrap original Gaussian Splatting models and facilitates both usage and integration with PyTorch3D.
    """
    def __init__(self, 
                 source_path: str,
                 output_path: str,
                 iteration_to_load:int=30_000,
                 model_params: ModelParams=None,
                 pipeline_params: PipelineParams=None,
                 opt_params: OptimizationParams=None,
                 load_gt_images=True,
                 eval_split=False,
                 eval_split_interval=8,
                 ) -> None:
        """Initialize the Gaussian Splatting model wrapper.
        
        Args:
            source_path (str): Path to the directory containing the source images.
            output_path (str): Path to the directory containing the output of the Gaussian Splatting optimization.
            iteration_to_load (int, optional): Checkpoint to load. Should be 7000 or 30_000. Defaults to 30_000.
            model_params (ModelParams, optional): Model parameters. Defaults to None.
            pipeline_params (PipelineParams, optional): Pipeline parameters. Defaults to None.
            opt_params (OptimizationParams, optional): Optimization parameters. Defaults to None.
            load_gt_images (bool, optional): If True, will load all GT images in the source folder.
                Useful for evaluating the model, but loading can take a few minutes. Defaults to True.
            eval_split (bool, optional): If True, will split images and cameras into a training set and an evaluation set. 
                Defaults to False.
            eval_split_interval (int, optional): Every eval_split_interval images, an image is added to the evaluation set. 
                Defaults to 8 (following standard practice).
        """
        self.source_path = source_path
        self.output_path = output_path
        self.loaded_iteration = iteration_to_load
        
        
        if model_params is None:
            model_params = ModelParams()
        if pipeline_params is None:
            pipeline_params = PipelineParams()
        if opt_params is None:
            opt_params = OptimizationParams()
        
        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.opt_params = opt_params

        
        self._C0 = 0.28209479177387814
        
        cam_list = load_gs_cameras(
            source_path=source_path,
            gs_output_path=output_path,
            load_gt_images=load_gt_images,
            )
        
        if eval_split:
            self.cam_list = []
            self.test_cam_list = []
            for i, cam in enumerate(cam_list):
                if i % eval_split_interval == 0:
                    self.test_cam_list.append(cam)
                else:
                    self.cam_list.append(cam)
            # test_ns_cameras = convert_camera_from_gs_to_nerfstudio(self.test_cam_list)
            # self.test_cameras = NeRFCameras.from_ns_cameras(test_ns_cameras)
            self.test_cameras = CamerasWrapper(self.test_cam_list)

        else:
            self.cam_list = cam_list
            self.test_cam_list = None
            self.test_cameras = None
            
        # ns_cameras = convert_camera_from_gs_to_nerfstudio(self.cam_list)
        # self.training_cameras = NeRFCameras.from_ns_cameras(ns_cameras)
        self.training_cameras = CamerasWrapper(self.cam_list)
            
        self.gaussians = GaussianModel(self.model_params.sh_degree)
        self.gaussians.load_ply(
            os.path.join(
                output_path,
                "point_cloud",
                "iteration_" + str(iteration_to_load),
                "point_cloud.ply"
                )
            )
        

    @property
    def device(self):
        with torch.no_grad():
            return self.gaussians.get_xyz.device
    
    @property
    def image_height(self):
        return self.cam_list[0].image_height
    
    @property
    def image_width(self):
        return self.cam_list[0].image_width
    
    def render_image(
        self,
        nerf_cameras:CamerasWrapper=None, 
        camera_indices:int=0,
        return_whole_package=False):
        """Render an image with Gaussian Splatting rasterizer.

        Args:
            nerf_cameras (CamerasWrapper, optional): Set of cameras. 
                If None, uses the training cameras, but can be any set of cameras. Defaults to None.
            camera_indices (int, optional): Index of the camera to render in the set of cameras. 
                Defaults to 0.
            return_whole_package (bool, optional): If True, returns the whole output package 
                as computed in the original rasterizer from 3D Gaussian Splatting paper. Defaults to False.

        Returns:
            Tensor or Dict: A tensor of the rendered RGB image, or the whole output package.
        """
        
        if nerf_cameras is None:
            gs_cameras = self.cam_list
        else:
            gs_cameras = nerf_cameras.gs_cameras
        
        camera = gs_cameras[camera_indices]
        render_pkg = gs_render(camera, self.gaussians, 
                            self.pipeline_params, 
                            bg_color=torch.zeros(3, device='cuda'))
        
        if return_whole_package:
            return render_pkg
        else:
            image = render_pkg["render"]
            return image.permute(1, 2, 0)
    
    def get_gt_image(self, camera_indices:int, to_cuda=False):
        """Returns the ground truth image corresponding to the training camera at the given index.

        Args:
            camera_indices (int): Index of the camera in the set of cameras.
            to_cuda (bool, optional): If True, moves the image to GPU. Defaults to False.

        Returns:
            Tensor: The ground truth image.
        """
        gt_image = self.cam_list[camera_indices].original_image
        if to_cuda:
            gt_image = gt_image.cuda()
        return gt_image.permute(1, 2, 0)
    
    def get_test_gt_image(self, camera_indices:int, to_cuda=False):
        """Returns the ground truth image corresponding to the test camera at the given index.
        
        Args:
            camera_indices (int): Index of the camera in the set of cameras.
            to_cuda (bool, optional): If True, moves the image to GPU. Defaults to False.
        
        Returns:
            Tensor: The ground truth image.
        """
        gt_image = self.test_cam_list[camera_indices].original_image
        if to_cuda:
            gt_image = gt_image.cuda()
        return gt_image.permute(1, 2, 0)
    
    def downscale_output_resolution(self, downscale_factor):
        """Downscale the output resolution of the Gaussian Splatting model.

        Args:
            downscale_factor (float): Factor by which to downscale the resolution.
        """
        self.training_cameras.rescale_output_resolution(1.0 / downscale_factor)
    
    def generate_point_cloud(self):
        """Generate a point cloud from the Gaussian Splatting model.

        Returns:
            (Tensor, Tensor): The points and the colors of the point cloud.
                Each has shape (N, 3), where N is the number of Gaussians.
        """
        with torch.no_grad():
            points = self.gaussians.get_xyz
            # colors = self.gaussians.get_features[:, 0] * self._C0 + 0.5
            colors = SH2RGB(self.gaussians.get_features[:, 0])
            
        return points, colors
    
    def plot_point_cloud(
        self,
        points=None,
        colors=None,
        n_points_to_plot: int = 50000,
        width=1000,
        height=500,
    ):
        """Plot the generated 3D point cloud with plotly.

        Args:
            n_points_to_plot (int, optional): _description_. Defaults to 50000.
            points (_type_, optional): _description_. Defaults to None.
            colors (_type_, optional): _description_. Defaults to None.
            width (int, optional): Defaults to 1000.
            height (int, optional): Defaults to 1000.

        Raises:
            ValueError: _description_

        Returns:
            go.Figure: The plotly figure.
        """
        
        with torch.no_grad():
            if points is None:
                points, colors = self.generate_point_cloud()

            points_idx = torch.randperm(points.shape[0])[:n_points_to_plot]
            points_to_plot = points[points_idx].cpu()
            colors_to_plot = colors[points_idx].cpu()

            z = points_to_plot[:, 2]
            x = points_to_plot[:, 0]
            y = points_to_plot[:, 1]
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=3,
                    color=colors_to_plot,  # set color to an array/list of desired values
                    # colorscale = 'Magma'
                ),
            )
            layout = go.Layout(
                scene=dict(bgcolor="white", aspectmode="data"),
                template="none",
                width=width,
                height=height,
            )
            fig = go.Figure(data=[trace], layout=layout)
            # fig.update_layout(template='none', scene_aspectmode='data')

            # fig.show()
            return fig

class HandGaussianSplattingWrapper(GaussianSplattingWrapper):
    """Class to wrap original Gaussian Splatting models and facilitates both usage and integration with PyTorch3D.
    """
    def __init__(self, 
                 source_path: str,
                 output_path: str,
                 mano_ply_path: str,
                 iteration_to_load:int=None,
                 model_params: ModelParams=None,
                 pipeline_params: PipelineParams=None,
                 opt_params: OptimizationParams=None,
                 load_gt_images=True,
                 eval_split=False,
                 eval_split_interval=8,
                 ) -> None:
        """Initialize the Gaussian Splatting model wrapper.
        
        Args:
            source_path (str): Path to the directory containing the source images.
            output_path (str): Path to the directory containing the output of the Gaussian Splatting optimization.
            iteration_to_load (int, optional): Checkpoint to load. Should be 7000 or 30_000. Defaults to 30_000.
            model_params (ModelParams, optional): Model parameters. Defaults to None.
            pipeline_params (PipelineParams, optional): Pipeline parameters. Defaults to None.
            opt_params (OptimizationParams, optional): Optimization parameters. Defaults to None.
            load_gt_images (bool, optional): If True, will load all GT images in the source folder.
                Useful for evaluating the model, but loading can take a few minutes. Defaults to True.
            eval_split (bool, optional): If True, will split images and cameras into a training set and an evaluation set. 
                Defaults to False.
            eval_split_interval (int, optional): Every eval_split_interval images, an image is added to the evaluation set. 
                Defaults to 8 (following standard practice).
        """
        self.source_path = source_path
        self.output_path = output_path
        #self.loaded_iteration = iteration_to_load
        
        
        if model_params is None:
            model_params = ModelParams()
        if pipeline_params is None:
            pipeline_params = PipelineParams()
        if opt_params is None:
            opt_params = OptimizationParams()
        
        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.opt_params = opt_params

        
        self._C0 = 0.28209479177387814
        
        cam_list = load_gs_cameras(
            source_path=source_path,
            gs_output_path=output_path,
            load_gt_images=load_gt_images,
            )
        
        if eval_split:
            self.cam_list = []
            self.test_cam_list = []
            for i, cam in enumerate(cam_list):
                if i % eval_split_interval == 0:
                    self.test_cam_list.append(cam)
                else:
                    self.cam_list.append(cam)
            # test_ns_cameras = convert_camera_from_gs_to_nerfstudio(self.test_cam_list)
            # self.test_cameras = NeRFCameras.from_ns_cameras(test_ns_cameras)
            self.test_cameras = CamerasWrapper(self.test_cam_list)

        else:
            self.cam_list = cam_list
            self.test_cam_list = None
            self.test_cameras = None
            
        # ns_cameras = convert_camera_from_gs_to_nerfstudio(self.cam_list)
        # self.training_cameras = NeRFCameras.from_ns_cameras(ns_cameras)
        self.training_cameras = CamerasWrapper(self.cam_list)
        
        self.gaussians = GaussianModel(self.model_params.sh_degree)
        self.initialize_from_mano_pcd(mano_ply_path)

    def initialize_from_mano_pcd(self, mano_ply_path):
        plydata = PlyData.read(mano_ply_path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        rgb = np.stack((np.asarray(plydata.elements[0]["red"]),
                        np.asarray(plydata.elements[0]["green"]),
                        np.asarray(plydata.elements[0]["blue"])),  axis=1)

        spatial_lr_scale = self.gaussians.spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(rgb)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.gaussians.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        #opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        opacities = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        self.gaussians._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.gaussians._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.gaussians._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.gaussians._scaling = nn.Parameter(scales.requires_grad_(True))
        self.gaussians._rotation = nn.Parameter(rots.requires_grad_(True))
        self.gaussians._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")
