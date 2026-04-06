from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from pytorch3d.transforms import quaternion_apply, quaternion_invert, quaternion_to_matrix, rotation_6d_to_matrix
from pytorch3d.io import save_obj

import numpy as np
import torch
import math
import os

from diff_gaussian_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings3dgs
from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer3dgs

from utils.sh_utils import RGB2SH, SH2RGB, eval_sh
from utils.camera_utils import load_gs_camera, nerfmodel_training_camera_json, CamerasWrapper, convert_camera_from_gs_to_pytorch3d

def get_points_rgb(
    gsmodel,
    positions:torch.Tensor=None,
    camera_centers:torch.Tensor=None,
    directions:torch.Tensor=None,
    sh_levels:int=None,
    sh_coordinates:torch.Tensor=None,
    ):
    """Returns the RGB color of the points for the given camera pose.

    Args:
        positions (torch.Tensor, optional): Shape (n_pts, 3). Defaults to None.
        camera_centers (torch.Tensor, optional): Shape (n_pts, 3) or (1, 3). Defaults to None.
        directions (torch.Tensor, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
        
    if positions is None:
        positions = gsmodel.points

    if camera_centers is not None:
        render_directions = torch.nn.functional.normalize(positions - camera_centers, dim=-1)
    elif directions is not None:
        render_directions = directions
    else:
        raise ValueError("Either camera_centers or directions must be provided.")

    if sh_coordinates is None:
        #sh_coordinates = gsmodel.sh_coordinates
        sh_coordinates = gsmodel.get_features
        
    if sh_levels is None:
        sh_coordinates = sh_coordinates
    else:
        sh_coordinates = sh_coordinates[:, :sh_levels**2]

    shs_view = sh_coordinates.transpose(-1, -2).view(-1, 3, sh_levels**2)
    sh2rgb = eval_sh(sh_levels-1, shs_view, render_directions)
    colors = torch.clamp_min(sh2rgb + 0.5, 0.0).view(-1, 3)
    
    return colors

def render_image_gaussian_rasterizer_pose(
    gsmodel, 
    pose=None, 
    sh_deg:int=None,
    return_colors:bool=False,
    ):
    """Render an image using the Gaussian Splatting Rasterizer.

    Args:
        nerf_cameras (CamerasWrapper, optional): _description_. Defaults to None.
        sh_deg (int, optional): _description_. Defaults to None.
        quaternions (_type_, optional): _description_. Defaults to None.
        use_same_scale_in_all_directions (bool, optional): _description_. Defaults to False.
        return_colors (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    #print(f'camera_indices: {camera_indices}, verbose: {verbose}, bg_color: {bg_color}, sh_deg: {sh_deg}') # 0, False, None, 0
    #print(f'quaternions is None: {quaternions is None}, use_same_scale_in_all_directions: {use_same_scale_in_all_directions}, point_colors is None: {point_colors is None}')# True, False, True

    viewpoint_camera = pose

    splat_opacities = gsmodel.get_opacity.view(-1, 1)

    device = splat_opacities.device

    bg_color = torch.Tensor([1.0, 1.0, 1.0]).to(device)
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) 
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5) 
    positions = gsmodel.get_xyz

    raster_settings = GaussianRasterizationSettings3dgs(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.,
        viewmatrix= viewpoint_camera.world_view_transform,
        projmatrix= viewpoint_camera.full_proj_transform,
        sh_degree=sh_deg,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
    )
    rasterizer = GaussianRasterizer3dgs(raster_settings=raster_settings)

    # TODO: Change color computation to match 3DGS paper (remove sigmoid)
    splat_colors = get_points_rgb(
        gsmodel,
        positions=positions, 
        camera_centers=viewpoint_camera.camera_center,
        sh_levels=sh_deg+1,)
    shs = None

    print(f'splat_colors.shape: {splat_colors.shape}')

    quaternions = gsmodel.get_rotation
    
    scales = gsmodel.get_scaling
    
    cov3D = None
    n_points = positions.shape[0]
    print(f'n_points: {n_points}')
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(self._points, dtype=self._points.dtype, requires_grad=True, device=self.device) + 0
    screenspace_points = torch.zeros(n_points, 3, dtype=positions.dtype, requires_grad=True, device=positions.device)

    means2D = screenspace_points
    
    rendered_image, radii, depth_alpha = rasterizer(
        means3D = positions,
        means2D = means2D,
        shs = shs,
        colors_precomp = splat_colors,
        opacities = splat_opacities,
        scales = scales,
        rotations = quaternions,
        cov3D_precomp = cov3D)
    
    fov_x_sugar = 1.027507346206077
    depth, alpha = torch.chunk(depth_alpha, 2)
    focal = 1/(2* math.tan(fov_x_sugar/2))
    disp = focal/(depth + (alpha * 10)+ 1e-5)

    try:
        min_d = disp[alpha <=0.1].min()
    except Exception:
        min_d = disp.min()
    
    disp = torch.clamp((disp-min_d) / (disp.max()-min_d), 0.0, 1.0)

    outputs = {
        "image": rendered_image.transpose(0, 1).transpose(1, 2),
        "render": rendered_image,
        "radii": radii,
        "viewspace_points": screenspace_points,
        "depth":disp,
        "alpha":alpha,
        "scales":scales,
        "opacities":splat_opacities,
    }
    if return_colors:
        outputs["colors"] = splat_colors

    return outputs

def extract_per_gs(gsmodel, surface_mesh, sh_coordinates, cov, dataset, opt, square_size=10, n_sh=1, device='cuda'):
    from pytorch3d.renderer import (
    AmbientLights,
    MeshRenderer,
    SoftPhongShader,
    )
    from pytorch3d.renderer.blending import BlendParams
    texture_with_gaussian_renders = True
    
    if square_size < 3:
        raise ValueError("square_size must be >= 3")
    
    verts = surface_mesh.verts_list()[0]
    faces = surface_mesh.faces_list()[0]
    faces_verts = verts[faces]

    print(f'verts.shape: {verts.shape}, faces.shape: {faces.shape}, faces_verts.shape: {faces_verts.shape}') # [20002, 3], [39999, 3], [39999, 3, 3]
    
    n_triangles = len(faces)
    #n_gaussians_per_triangle = rc.n_gaussians_per_surface_triangle
    n_gaussians_per_triangle = 3
    n_squares = n_triangles // 2 + 1
    n_square_per_axis = int(np.sqrt(n_squares) + 1)
    texture_size = square_size * (n_square_per_axis)

    print(f'n_triangles: {n_triangles}, n_gaussians_per_triangle: {n_gaussians_per_triangle}, n_squares: {n_squares}, n_square_per_axis: {n_square_per_axis}, texture_size: {texture_size}') # 39999, 55, 20000, 142, 1420
    print(f'sh_coordinates.shape: {sh_coordinates.shape}')
    #faces_features = sh_coordinates[:, :n_sh].reshape(n_triangles, n_gaussians_per_triangle, n_sh * 3)
    #faces_features = sh_coordinates.reshape(n_triangles, n_gaussians_per_triangle, n_sh * 3)
    faces_features = sh_coordinates[:, :n_sh].reshape(n_triangles, n_gaussians_per_triangle, n_sh * 3)
    n_features = faces_features.shape[-1]
    
    if texture_with_gaussian_renders:
        n_features = 3

    print(f'faces_features.shape: {faces_features.shape}, n_features: {n_features}') # [596991, 3, 3], 3
    
    # Build faces UV.
    # Each face will have 3 corresponding vertices in the UV map
    faces_uv = torch.arange(3 * n_triangles, device=device).view(n_triangles, 3)  # n_triangles, 3
    
    # Build corresponding vertices UV
    vertices_uv = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=device), 
        torch.arange(n_square_per_axis, device=device))
    bottom_verts_uv = torch.cat(
        [vertices_uv[n_square_per_axis:-1, None], vertices_uv[:-n_square_per_axis-1, None], vertices_uv[n_square_per_axis+1:, None]],
        dim=1)
    top_verts_uv = torch.cat(
        [vertices_uv[1:-n_square_per_axis, None], vertices_uv[:-n_square_per_axis-1, None], vertices_uv[n_square_per_axis+1:, None]],
        dim=1)
    
    vertices_uv = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=device), 
        torch.arange(n_square_per_axis, device=device))[:, None]
    u_shift = torch.tensor([[1, 0]], dtype=torch.int32, device=device)[:, None]
    v_shift = torch.tensor([[0, 1]], dtype=torch.int32, device=device)[:, None]
    bottom_verts_uv = torch.cat(
        [vertices_uv + u_shift, vertices_uv, vertices_uv + u_shift + v_shift],
        dim=1)
    top_verts_uv = torch.cat(
        [vertices_uv + v_shift, vertices_uv, vertices_uv + u_shift + v_shift],
        dim=1)
    
    verts_uv = torch.cat([bottom_verts_uv, top_verts_uv], dim=1)
    verts_uv = verts_uv * square_size
    verts_uv[:, 0] = verts_uv[:, 0] + torch.tensor([[-2, 1]], device=device)
    verts_uv[:, 1] = verts_uv[:, 1] + torch.tensor([[2, 1]], device=device)
    verts_uv[:, 2] = verts_uv[:, 2] + torch.tensor([[-2, -3]], device=device)
    verts_uv[:, 3] = verts_uv[:, 3] + torch.tensor([[1, -1]], device=device)
    verts_uv[:, 4] = verts_uv[:, 4] + torch.tensor([[1, 3]], device=device)
    verts_uv[:, 5] = verts_uv[:, 5] + torch.tensor([[-3, -1]], device=device)
    verts_uv = verts_uv.reshape(-1, 2) / texture_size

    print(f'verts_uv.shape: {verts_uv.shape}') # [120984, 2]
    
    # ---Build texture image
    # Start by computing pixel indices for each triangle
    texture_img = torch.zeros(texture_size, texture_size, n_features, device=device)    
    pixel_idx_inside_bottom_triangle = torch.zeros(0, 2, dtype=torch.int32, device=device)
    pixel_idx_inside_top_triangle = torch.zeros(0, 2, dtype=torch.int32, device=device)
    for tri_i in range(0, square_size-1):
        for tri_j in range(0, tri_i+1):
            pixel_idx_inside_bottom_triangle = torch.cat(
                [pixel_idx_inside_bottom_triangle, torch.tensor([[tri_i, tri_j]], dtype=torch.int32, device=device)], dim=0)
    for tri_i in range(0, square_size):
        for tri_j in range(tri_i+1, square_size):
            pixel_idx_inside_top_triangle = torch.cat(
                [pixel_idx_inside_top_triangle, torch.tensor([[tri_i, tri_j]], dtype=torch.int32, device=device)], dim=0)
    
    bottom_triangle_pixel_idx = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=device), 
        torch.arange(n_square_per_axis, device=device))[:, None] * square_size + pixel_idx_inside_bottom_triangle[None]
    top_triangle_pixel_idx = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=device), 
        torch.arange(n_square_per_axis, device=device))[:, None] * square_size + pixel_idx_inside_top_triangle[None]
    triangle_pixel_idx = torch.cat(
        [bottom_triangle_pixel_idx[:, None], 
        top_triangle_pixel_idx[:, None]],
        dim=1).view(-1, bottom_triangle_pixel_idx.shape[-2], 2)[:n_triangles]
    
    # Then we compute the barycentric coordinates of each pixel inside its corresponding triangle
    bottom_triangle_pixel_bary_coords = pixel_idx_inside_bottom_triangle.clone().float()
    bottom_triangle_pixel_bary_coords[..., 0] = -(bottom_triangle_pixel_bary_coords[..., 0] - (square_size - 2))
    bottom_triangle_pixel_bary_coords[..., 1] = (bottom_triangle_pixel_bary_coords[..., 1] - 1)
    bottom_triangle_pixel_bary_coords = (bottom_triangle_pixel_bary_coords + 0.) / (square_size - 3)
    bottom_triangle_pixel_bary_coords = torch.cat(
        [1. - bottom_triangle_pixel_bary_coords.sum(dim=-1, keepdim=True), bottom_triangle_pixel_bary_coords],
        dim=-1)
    top_triangle_pixel_bary_coords = pixel_idx_inside_top_triangle.clone().float()
    top_triangle_pixel_bary_coords[..., 0] = (top_triangle_pixel_bary_coords[..., 0] - 1)
    top_triangle_pixel_bary_coords[..., 1] = -(top_triangle_pixel_bary_coords[..., 1] - (square_size - 1))
    top_triangle_pixel_bary_coords = (top_triangle_pixel_bary_coords + 0.) / (square_size - 3)
    top_triangle_pixel_bary_coords = torch.cat(
        [1. - top_triangle_pixel_bary_coords.sum(dim=-1, keepdim=True), top_triangle_pixel_bary_coords],
        dim=-1)
    triangle_pixel_bary_coords = torch.cat(
        [bottom_triangle_pixel_bary_coords[None],
        top_triangle_pixel_bary_coords[None]],
        dim=0)  # 2, n_pixels_per_triangle, 3
    
    all_triangle_bary_coords = triangle_pixel_bary_coords[None].expand(n_squares, -1, -1, -1).reshape(-1, triangle_pixel_bary_coords.shape[-2], 3)
    all_triangle_bary_coords = all_triangle_bary_coords[:len(faces_verts)]
    
    pixels_space_positions = (all_triangle_bary_coords[..., None] * faces_verts[:, None]).sum(dim=-2)[:, :, None]
    
    #gaussian_centers = rc.points.reshape(-1, 1, n_gaussians_per_triangle, 3)
    gaussian_centers = faces_verts.reshape(-1, 1, n_gaussians_per_triangle, 3)
    #gaussian_inv_scaled_rotation = rc.get_covariance(return_full_matrix=True, return_sqrt=True, inverse_scales=True).reshape(-1, 1, rc.n_gaussians_per_surface_triangle, 3, 3)
    gaussian_inv_scaled_rotation = cov.reshape(-1, 1, n_gaussians_per_triangle, 3, 3)
    
    print(f'all_triangle_bary_coords.shape: {all_triangle_bary_coords.shape}') # [39999, 45, 3]
    print(f'pixels_space_positions.shape: {pixels_space_positions.shape}') # [39999, 45, 1, 3]
    print(f'gaussian_centers.shape: {gaussian_centers.shape}') # [39999, 1, 55, 3]
    print(f'cov.shape: {cov.shape}, gaussian_inv_scaled_rotation.shape: {gaussian_inv_scaled_rotation.shape}') # [39999, 1, 55, 3, 3]

    # Compute the density field as a sum of local gaussian opacities
    shift = (pixels_space_positions - gaussian_centers)
    warped_shift = gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
    neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
    neighbor_opacities = torch.exp(-1. / 2 * neighbor_opacities)
    
    print(f'neighbor_opacities.shape: {neighbor_opacities.shape}') # [801455, 45, 3]

    pixel_features = faces_features[:, None].expand(-1, neighbor_opacities.shape[1], -1, -1).gather(
        dim=-2,
        index=neighbor_opacities[..., None].argmax(dim=-2, keepdim=True).expand(-1, -1, -1, 3)
        )[:, :, 0, :]
        
    # pixel_alpha = neighbor_opacities.sum(dim=-1, keepdim=True)
    texture_img[(triangle_pixel_idx[..., 0], triangle_pixel_idx[..., 1])] = pixel_features

    texture_img = texture_img.transpose(0, 1)
    texture_img = SH2RGB(texture_img.flip(0))
    
    faces_per_pixel = 1
    max_faces_per_bin = 50_000
    mesh_raster_settings = RasterizationSettings(
        image_size=(1024, 1024),
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel,
        # max_faces_per_bin=max_faces_per_bin
    )
    lights = AmbientLights(device=device)

    cam_list = load_gs_camera(nerfmodel_training_camera_json)
    train_cam_list = []
    test_cam_list = []
    eval_split_interval = 8
    for i, cam in enumerate(cam_list):
        if i % eval_split_interval == 0:
            test_cam_list.append(cam)
        else:
            train_cam_list.append(cam)
    training_cameras = CamerasWrapper(cam_list)
    p3d_cameras = training_cameras.p3d_cameras

    print(f'len(p3d_cameras): {len(p3d_cameras)}')

    rasterizer = MeshRasterizer(
            cameras=p3d_cameras[0], 
            raster_settings=mesh_raster_settings,
    )
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(
            device=device, 
            cameras=p3d_cameras[0],
            lights=lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
        )
    )
    texture_idx = torch.cartesian_prod(
        torch.arange(texture_size, device=device), 
        torch.arange(texture_size, device=device)
        ).reshape(texture_size, texture_size, 2
                    )
    texture_idx = torch.cat([texture_idx, torch.zeros_like(texture_idx[..., 0:1])], dim=-1)
    texture_counter = torch.zeros(texture_size, texture_size, 1, device=device)
    idx_textures_uv = TexturesUV(
        maps=texture_idx[None].float(), #texture_img[None]),
        verts_uvs=verts_uv[None],
        faces_uvs=faces_uv[None],
        sampling_mode='bilinear',
        )
    idx_mesh = Meshes(
        verts=[surface_mesh.verts_list()[0]],   
        faces=[surface_mesh.faces_list()[0]],
        textures=idx_textures_uv,
        )
    
    print(f'texture_idx.shape: {texture_idx.shape}') # [1420, 1420, 3]
    print(f'texture_counter.shape: {texture_counter.shape}') # [1420, 1420, 1]
    #print(f'idx_textures_uv.shape: {idx_textures_uv.shape}')
    #print(f'idx_mesh.shape: {idx_mesh.shape}')
    
    for cam_idx in range(len(training_cameras)):
        p3d_camera = p3d_cameras[cam_idx]
        
        # Render rgb img
        rgb_img_out = render_image_gaussian_rasterizer_pose(
            gsmodel,
            pose=training_cameras.gs_cameras[cam_idx],
            sh_deg=gsmodel.active_sh_degree,
        )
        rgb_img = rgb_img_out['image'].clamp(min = 0, max = 1)
        
        fragments = renderer.rasterizer(idx_mesh, cameras=p3d_camera)
        idx_img = renderer.shader(fragments, idx_mesh, cameras=p3d_camera)[0, ..., :2]
        # print("Idx img:", idx_img.shape, idx_img.min(), idx_img.max())
        update_mask = fragments.zbuf[0, ..., 0] > 0
        idx_to_update = idx_img[update_mask].round().long() 

        no_initialize_mask = texture_counter[(idx_to_update[..., 0], idx_to_update[..., 1])][..., 0] != 0
        texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] = no_initialize_mask[..., None] * texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])]

        texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] = texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] + rgb_img[update_mask]
        texture_counter[(idx_to_update[..., 0], idx_to_update[..., 1])] = texture_counter[(idx_to_update[..., 0], idx_to_update[..., 1])] + 1

    texture_img = texture_img / texture_counter.clamp(min=1)        
    
    return verts_uv, faces_uv, texture_img

def extract_texture_from_hoi_gs(hoi_gaussians, dataset, opt, mesh_save_path):
    xyz = hoi_gaussians.get_xyz
    n_obj_gs = hoi_gaussians.obj_gaussians._xyz.shape[0]
    #obj_xyz, hand_xyz = xyz[:n_obj_gs], xyz[n_obj_gs:]
    device = xyz.device

    obj_tri = hoi_gaussians.obj_gaussians.faces
    hand_tri = hoi_gaussians.hand_gaussians.mano_layer.faces.astype(int)
    hand_tri = torch.from_numpy(hand_tri).to(device)

    hand_tri += n_obj_gs

    hoi_tri = torch.cat((obj_tri, hand_tri), dim=0)

    n_sh = 1
    sh_deg = dataset.sh_degree
    print(f'sh_deg: {sh_deg}')
    features = hoi_gaussians.get_features
    #sh_coordinates = xyz[hoi_tri]
    sh_coordinates = features[hoi_tri].reshape(-1, (sh_deg+1)**2, 3)
    shs_view = features.transpose(-1,-2)
    #n_sh = features.shape[1]
    
    cov = rotation_6d_to_matrix(hoi_gaussians.get_covariance())
    cov = cov[hoi_tri]
    
    hoi_surf_mesh = Meshes(
            verts=[xyz],   
            faces=[hoi_tri.to(device)])
    hoi_vert_normals = hoi_surf_mesh.verts_normals_list()[0]

    print(f'sh_coordinates.shape: {sh_coordinates.shape}, shs_view.shape: {shs_view.shape}, n_sh: {n_sh}')
    print(f'xyz.shape: {xyz.shape}, hoi_tri.shape: {hoi_tri.shape}, hoi_vert_normals.shape: {hoi_vert_normals.shape}')

    render_directions = quaternion_apply(quaternion_invert(hoi_gaussians.get_rotation), hoi_vert_normals).view(-1, 3)
    v_rgb = eval_sh(sh_deg, shs_view, render_directions)

    print(f'v_rgb.shape: {v_rgb.shape}, cov.shape: {cov.shape}')

    hoi_surf_mesh = Meshes(
            verts=[xyz],   
            faces=[hoi_tri.to(device)],
            textures=TexturesVertex([v_rgb]),
            # verts_normals=[verts_normals.to(device)],
            )

    with torch.no_grad():
        verts_uv, faces_uv, texture_img = extract_per_gs(hoi_gaussians, hoi_surf_mesh, sh_coordinates, cov, dataset, opt, 
                                            square_size=10, n_sh=n_sh, device=device)

        textures_uv = TexturesUV(
            maps=texture_img[None], #texture_img[None]),
            verts_uvs=verts_uv[None],
            faces_uvs=faces_uv[None],
            sampling_mode='bilinear',
            )
        textured_mesh = Meshes(
            verts=[hoi_surf_mesh.verts_list()[0]],   
            faces=[hoi_surf_mesh.faces_list()[0]],
            textures=textures_uv,
            )
        print(f'mesh_save_path: {mesh_save_path}')
        save_obj(  
            mesh_save_path,
            verts=textured_mesh.verts_list()[0],
            faces=textured_mesh.faces_list()[0],
            verts_uvs=textured_mesh.textures.verts_uvs_list()[0],
            faces_uvs=textured_mesh.textures.faces_uvs_list()[0],
            texture_map=textured_mesh.textures.maps_padded()[0].clamp(0., 1.),
            )
    
    return verts_uv, faces_uv, texture_img