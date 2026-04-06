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

from scene.cameras import Camera, RCamera
import torch
import numpy as np

from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix

from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, getProjectionMatrix

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def loadRandomCam(opt, id, cam_info, resolution_scale, SSAA=False):
    return RCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, delta_polar=cam_info.delta_polar,
                  delta_azimuth=cam_info.delta_azimuth , delta_radius=cam_info.delta_radius, opt=opt, 
                  uid=id, data_device=opt.device, SSAA=SSAA)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def cameraList_from_RcamInfos(cam_infos, resolution_scale, opt, SSAA=False):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadRandomCam(opt, id, c, resolution_scale, SSAA=SSAA))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : id,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

# SuGaR cams
nerfmodel_training_camera_json = [
    {"id": 0, "img_name": 0, "width": 512, "height": 512, 
        "position": [-0.0, 2.4748739450959736, -2.474873945095974], 
        "rotation": [[-1.0, -0.0, -0.0], [-0.0, -0.7071068528928167, -0.7071067932881648], 
        [-0.0, -0.7071068528928167, 0.7071067932881648]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 1, "img_name": 1, "width": 512, "height": 512, 
        "position": [1.7500000205439297, 1.75000002054393, -2.474873747455072], 
        "rotation": [[-0.7071067932881648, -0.5000000149011616, -0.5000000149011616], 
        [0.7071067932881648, -0.5000000149011616, -0.5000000149011616], 
        [-4.700437341686189e-19, -0.7071068143615901, 0.7071067722147396]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 2, "img_name": 2, "width": 512, "height": 512, 
        "position": [2.4748739450959687, -1.0818018879400336e-07, -2.474873945095974], 
        "rotation": [[4.3711389200881713e-08, -0.7071068528928153, -0.7071067932881635], 
        [0.999999999999998, 3.0908625192815224e-08, 3.0908621640101154e-08], 
        [-6.698506759908611e-16, -0.7071068528928167, 0.7071067932881648]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 3, "img_name": 3, "width": 512, "height": 512, 
        "position": [1.7500000205439297, -1.75000002054393, -2.474873747455072], 
        "rotation": [[0.7071067932881648, -0.5000000149011616, -0.5000000149011616], 
        [0.7071067932881648, 0.5000000149011616, 0.5000000149011616], 
        [4.700437341686189e-19, -0.7071068143615901, 0.7071067722147396]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 4, "img_name": 4, "width": 512, "height": 512, 
        "position": [-2.1636037758800554e-07, -2.4748739450959545, -2.474873945095974], 
        "rotation": [[0.9999999999999923, 6.181725038563009e-08, 6.181724328020195e-08], 
        [-8.742277840176292e-08, 0.7071068528928113, 0.7071067932881594], 
        [-1.3397013494215794e-15, -0.7071068528928167, 0.7071067932881648]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 5, "img_name": 5, "width": 512, "height": 512, 
        "position": [-1.7500003558011703, -1.749999683788923, -2.4748738096163603], 
        "rotation": [[0.7071066668475958, 0.5000000557696764, 0.500000112817687], 
        [-0.7071069197286912, 0.4999999020834427, 0.4999999293291286], 
        [1.0712716569701237e-14, -0.7071068652373388, 0.7071067809436367]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 6, "img_name": 6, "width": 512, "height": 512, 
        "position": [-2.474873945095973, 2.951258067672527e-08, -2.474873945095974], 
        "rotation": [[-1.1924881820086377e-08, 0.7071068528928165, 0.7071067932881647], 
        [-0.9999999999999998, -8.432165536151482e-09, -8.432164647972959e-09], 
        [5.025962168241836e-16, -0.7071068528928167, 0.7071067932881648]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 7, "img_name": 7, "width": 512, "height": 512, 
        "position": [-1.7499993224543853, 1.7500005621707697, -2.4748737105765803], 
        "rotation": [[-0.707107077779297, 0.4999998239619573, 0.49999982396195736], 
        [-0.7071065087968408, -0.5000001611367461, -0.5000001611367462], 
        [-1.490114363401331e-08, -0.7071068248983047, 0.7071067616780248]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 8, "img_name": 8, "width": 512, "height": 512, 
        "position": [-0.0, 2.4748739450959736, 2.474873945095974], 
        "rotation": [[-1.0, -0.0, -0.0], [-0.0, 0.7071068528928167, -0.7071067932881648], 
        [-0.0, -0.7071068528928167, -0.7071067932881648]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 9, "img_name": 9, "width": 512, "height": 512, 
        "position": [1.7500000205439297, 1.75000002054393, 2.474873747455072], 
        "rotation": [[-0.7071067932881648, 0.5000000149011616, -0.5000000149011616], 
        [0.7071067932881648, 0.5000000149011616, -0.5000000149011616], 
        [4.700437341686189e-19, -0.7071068143615901, -0.7071067722147396]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 10, "img_name": 10, "width": 512, "height": 512, 
        "position": [2.4748739450959687, -1.0818018879400336e-07, 2.474873945095974], 
        "rotation": [[4.3711389200881713e-08, 0.7071068528928153, -0.7071067932881635], 
        [0.999999999999998, -3.0908625192815224e-08, 3.0908621640101154e-08], 
        [6.698506759908611e-16, -0.7071068528928167, -0.7071067932881648]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 11, "img_name": 11, "width": 512, "height": 512, 
        "position": [1.7500000205439297, -1.75000002054393, 2.474873747455072], 
        "rotation": [[0.7071067932881648, 0.5000000149011616, -0.5000000149011616], 
        [0.7071067932881648, -0.5000000149011616, 0.5000000149011616], 
        [-4.700437341686189e-19, -0.7071068143615901, -0.7071067722147396]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 12, "img_name": 12, "width": 512, "height": 512, 
        "position": [-2.1636037758800554e-07, -2.4748739450959545, 2.474873945095974], 
        "rotation": [[0.9999999999999923, -6.181725038563009e-08, 6.181724328020195e-08], 
        [-8.742277840176292e-08, -0.7071068528928113, 0.7071067932881594], 
        [1.3397013494215794e-15, -0.7071068528928167, -0.7071067932881648]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 13, "img_name": 13, "width": 512, "height": 512, 
        "position": [-1.7500003558011703, -1.749999683788923, 2.4748738096163603], 
        "rotation": [[0.7071066668475958, -0.5000000557696764, 0.500000112817687], 
        [-0.7071069197286912, -0.4999999020834427, 0.4999999293291286], 
        [-1.0712716569701237e-14, -0.7071068652373388, -0.7071067809436367]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 14, "img_name": 14, "width": 512, "height": 512, 
        "position": [-2.474873945095973, 2.951258067672527e-08, 2.474873945095974], 
        "rotation": [[-1.1924881820086377e-08, -0.7071068528928165, 0.7071067932881647], 
        [-0.9999999999999998, 8.432165536151482e-09, -8.432164647972959e-09], 
        [-5.025962168241836e-16, -0.7071068528928167, -0.7071067932881648]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}, 
    {"id": 15, "img_name": 15, "width": 512, "height": 512, 
        "position": [-1.7499993224543853, 1.7500005621707697, 2.4748737105765803], 
        "rotation": [[-0.707107077779297, -0.4999998239619573, 0.49999982396195736], 
        [-0.7071065087968408, 0.5000001611367461, -0.5000001611367462], 
        [1.490114363401331e-08, -0.7071068248983047, -0.7071067616780248]], 
        "fy": 907.3232545158436, "fx": 907.3232545158436}]

def load_gs_camera(unsorted_camera_transforms):
    camera_transforms = sorted(unsorted_camera_transforms.copy(), key = lambda x : x['img_name'])

    cam_list = []
    for cam_idx in range(len(camera_transforms)):
        camera_transform = camera_transforms[cam_idx]
        
        # Extrinsics
        rot = np.array(camera_transform['rotation'])
        pos = np.array(camera_transform['position'])
        
        W2C = np.zeros((4,4))
        W2C[:3, :3] = rot
        W2C[:3, 3] = pos
        W2C[3,3] = 1
        
        Rt = np.linalg.inv(W2C)
        T = Rt[:3, 3]
        R = Rt[:3, :3].transpose()
        
        # Intrinsics
        width = 1024
        height = 1024
        fy = camera_transform['fy']
        fx = camera_transform['fx']
        fov_y = focal2fov(fy, height)
        fov_x = focal2fov(fx, width)
        
        gs_camera = GSCamera(
            colmap_id=camera_transform['id'], gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y, uid=camera_transform['id'],
            image_height=height, image_width=width,)
        
        cam_list.append(gs_camera)

    return cam_list


class GSCamera(torch.nn.Module):
    """Class to store Gaussian Splatting camera parameters.
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, gt_alpha_mask,uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_height=None, image_width=None,
                 ):
        """
        Args:
            colmap_id (int): ID of the camera in the COLMAP reconstruction.
            R (np.array): Rotation matrix.
            T (np.array): Translation vector.
            FoVx (float): Field of view in the x direction.
            FoVy (float): Field of view in the y direction.
            image (np.array): GT image.
            gt_alpha_mask (_type_): _description_
            image_name (_type_): _description_
            uid (_type_): _description_
            trans (_type_, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
            scale (float, optional): _description_. Defaults to 1.0.
            data_device (str, optional): _description_. Defaults to "cuda".
            image_height (_type_, optional): _description_. Defaults to None.
            image_width (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super(GSCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image_height is None or  image_width is None:
            raise ValueError("Either image_height or image_width must be specified")
        else:
            self.image_height = image_height
            self.image_width = image_width


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    @property
    def device(self):
        return self.world_view_transform.device
    
    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self

class CamerasWrapper:
    """Class to wrap Gaussian Splatting camera parameters 
    and facilitates both usage and integration with PyTorch3D.
    """
    def __init__(
        self,
        gs_cameras,
        p3d_cameras=None,
        p3d_cameras_computed=False,
    ) -> None:
        """
        Args:
            camera_to_worlds (_type_): _description_
            fx (_type_): _description_
            fy (_type_): _description_
            cx (_type_): _description_
            cy (_type_): _description_
            width (_type_): _description_
            height (_type_): _description_
            distortion_params (_type_): _description_
            camera_type (_type_): _description_
        """

        self.gs_cameras = gs_cameras
        
        self._p3d_cameras = p3d_cameras
        self._p3d_cameras_computed = p3d_cameras_computed
        
        device = gs_cameras[0].device        
        N = len(gs_cameras)
        R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
        T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
        self.fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
        self.fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
        self.height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.cx = self.width / 2.  # torch.zeros_like(fx).to(device)
        self.cy = self.height / 2.  # torch.zeros_like(fy).to(device)
        
        w2c = torch.zeros(N, 4, 4).to(device)
        w2c[:, :3, :3] = R.transpose(-1, -2)
        w2c[:, :3, 3] = T
        w2c[:, 3, 3] = 1
        
        c2w = w2c.inverse()
        c2w[:, :3, 1:3] *= -1
        c2w = c2w[:, :3, :]
        self.camera_to_worlds = c2w

    @classmethod
    def from_p3d_cameras(
        cls,
        p3d_cameras,
        width: float,
        height: float,
    ) -> None:
        """Initializes CamerasWrapper from pytorch3d-compatible camera object.

        Args:
            p3d_cameras (_type_): _description_
            width (float): _description_
            height (float): _description_

        Returns:
            _type_: _description_
        """
        cls._p3d_cameras = p3d_cameras
        cls._p3d_cameras_computed = True

        gs_cameras = convert_camera_from_pytorch3d_to_gs(
            p3d_cameras,
            height=height,
            width=width,
        )

        return cls(
            gs_cameras=gs_cameras,
            p3d_cameras=p3d_cameras,
            p3d_cameras_computed=True,
        )

    @property
    def device(self):
        return self.camera_to_worlds.device

    @property
    def p3d_cameras(self):
        if not self._p3d_cameras_computed:
            self._p3d_cameras = convert_camera_from_gs_to_pytorch3d(
                self.gs_cameras,
            )
            self._p3d_cameras_computed = True

        return self._p3d_cameras

    def __len__(self):
        return len(self.gs_cameras)

    def to(self, device):
        self.camera_to_worlds = self.camera_to_worlds.to(device)
        self.fx = self.fx.to(device)
        self.fy = self.fy.to(device)
        self.cx = self.cx.to(device)
        self.cy = self.cy.to(device)
        self.width = self.width.to(device)
        self.height = self.height.to(device)
        
        for gs_camera in self.gs_cameras:
            gs_camera.to(device)

        if self._p3d_cameras_computed:
            self._p3d_cameras = self._p3d_cameras.to(device)

        return self
        
    def get_spatial_extent(self):
        """Returns the spatial extent of the cameras, computed as 
        the extent of the bounding box containing all camera centers.

        Returns:
            (float): Spatial extent of the cameras.
        """
        camera_centers = self.p3d_cameras.get_camera_center()
        avg_camera_center = camera_centers.mean(dim=0, keepdim=True)
        half_diagonal = torch.norm(camera_centers - avg_camera_center, dim=-1).max().item()

        radius = 1.1 * half_diagonal
        return radius


def convert_camera_from_gs_to_pytorch3d(gs_cameras, device='cuda'):
    """
    From Gaussian Splatting camera parameters,
    computes R, T, K matrices and outputs pytorch3d-compatible camera object.

    Args:
        gs_cameras (List of GSCamera): List of Gaussian Splatting cameras.
        device (_type_, optional): _description_. Defaults to 'cuda'.

    Returns:
        p3d_cameras: pytorch3d-compatible camera object.
    """
    
    N = len(gs_cameras)
    
    R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
    T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
    fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
    fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
    image_height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    image_width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    cx = image_width / 2.  # torch.zeros_like(fx).to(device)
    cy = image_height / 2.  # torch.zeros_like(fy).to(device)
    
    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1
    
    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]
    
    distortion_params = torch.zeros(N, 6).to(device)
    camera_type = torch.ones(N, 1, dtype=torch.int32).to(device)

    # Pytorch3d-compatible camera matrices
    # Intrinsics
    image_size = torch.Tensor(
        [image_width[0], image_height[0]],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    p0_pytorch3d = (
        -(
            torch.Tensor(
                (cx[0], cy[0]),
            )[
                None
            ].to(device)
            - c0
        )
        / scale
    )
    focal_pytorch3d = (
        torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    )
    K = _get_sfm_calibration_matrix(
        1, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    K = K.expand(N, -1, -1)

    # Extrinsics
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.Tensor([-1.0, 1.0, -1]).to(device)
    T = T.squeeze(2)[:, :3] * torch.Tensor([-1.0, 1.0, -1]).to(device)

    p3d_cameras = P3DCameras(device=device, R=R, T=T, K=K, znear=0.0001)

    return p3d_cameras
