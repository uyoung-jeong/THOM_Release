import os
import os.path as osp
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import open3d as o3d
import torch
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from pytorch3d.ops import sample_farthest_points
from datetime import datetime

import sys
stage2_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(osp.dirname(__file__)))), 'stage2')
sys.path.append(stage2_dir)
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.spherical_harmonics import SH2RGB
from sugar_scene.gs_model import ModelParams

import copy


def extract_mesh_upsample(opt, lp, gs_model, mesh_path, 
                            xyz=None, features_dc=None, features_rest=None,
                            opacity=None, scale=None, rotation=None,
                            decimate=True, depth=None):
    # iniitialize sugar model
    # ====================Parameters====================
    device = gs_model._opacity.device

    # -----Model parameters-----
    use_train_test_split = True
    n_skip_images_for_eval_split = 1000000

    freeze_gaussians = False
    start_pruning_threshold = 0.1
    learnable_positions = True  # True in 3DGS
    use_same_scale_in_all_directions = False  # Should be False
        
    triangle_scale=2. # radiance mesh
    compute_color_in_rasterizer = True # rendering parameter

    # Regularization
    beta_mode = 'average'  # 'learnable', 'average' or 'weighted_average'
    density_factor = 1.

    regularize = True
    surface_level_knn_to_track = 16  # 8 until now

    #sh_levels = gs_model.active_sh_degree+1  # nerfmodel.gaussians.active_sh_degree + 1
    sh_levels = lp.sh_degree

    # ====================End of parameters====================
    checkpoint_dir = lp._model_path
    source_path = checkpoint_dir
    iteration_to_load = opt.meshify_iter
    gs_checkpoint_path = checkpoint_dir
    
    # ====================Load NeRF model and training data====================

    # Load Gaussian Splatting checkpoint 
    model_params = ModelParams()
    model_params.sh_degree = sh_levels
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        model_params=model_params,
        load_gt_images=False,
        eval_split=use_train_test_split,
        eval_split_interval=n_skip_images_for_eval_split,
        )
    
    if xyz is not None: # use this for coarse mesh extraction
        nerfmodel.gaussians._xyz = xyz
        nerfmodel.gaussians._features_dc = features_dc
        nerfmodel.gaussians._features_rest = features_rest
        nerfmodel.gaussians._opacity = opacity
        nerfmodel.gaussians._scaling = scale
        nerfmodel.gaussians._rotation = rotation
    
    # Point cloud
    with torch.no_grad():
        with torch.no_grad():
            sh_levels = int(np.sqrt(nerfmodel.gaussians.get_features.shape[1]))
            print(f"sh_levels: {sh_levels}")
        
        points = nerfmodel.gaussians.get_xyz.detach().float().cuda()
        colors = SH2RGB(nerfmodel.gaussians.get_features[:, 0].detach().float().cuda())
        n_points = len(points)

    # Mesh to bind to if needed  TODO
    o3d_mesh = None
    learn_surface_mesh_positions = False
    learn_surface_mesh_opacity = False
    learn_surface_mesh_scales = False
    n_gaussians_per_surface_triangle=1
    
    # ====================Initialize SuGaR model====================
    # Construct SuGaR model
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=points, #nerfmodel.gaussians.get_xyz.data,
        colors=colors, #0.5 + _C0 * nerfmodel.gaussians.get_features.data[:, 0, :],
        initialize=True,
        sh_levels=sh_levels,
        learnable_positions=learnable_positions,
        triangle_scale=triangle_scale,
        keep_track_of_knn=regularize,
        knn_to_track=surface_level_knn_to_track,
        beta_mode=beta_mode,
        freeze_gaussians=freeze_gaussians,
        surface_mesh_to_bind=o3d_mesh,
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=learn_surface_mesh_positions,
        learn_surface_mesh_opacity=learn_surface_mesh_opacity,
        learn_surface_mesh_scales=learn_surface_mesh_scales,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        )
    
    with torch.no_grad():            
        sugar._scales[...] = nerfmodel.gaussians._scaling.detach()
        sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()
        sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()
        sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()
        sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()
    
    ########## mesh extraction ##########
    # ========== Parameters ==========
    low_opacity_gaussian_pruning_threshold = 0.5

    # Surface level extraction parameters
    n_total_points = 10_000_000
    use_gaussian_depth_for_surface_levels = False  # False until now
    surface_level_triangle_scale = 2.  # 2.
    surface_level_primitive_types = 'diamond'  # 'diamond'
    surface_level_splat_mesh = True  # True
    surface_level_n_points_in_range = 21  # 21
    surface_level_range_size = 3.0  # 3.0
    surface_level_n_points_per_pass = 2_000_000  # '2_000_000'
    flat_surface_level_normals = False  # False
    use_fast_method = True  # TODO: Was False before, but True seems better

    # Mesh computation parameters
    fg_bbox_factor = 0.9  # 1.
    bg_bbox_factor = 4.  # 4.
    #poisson_depth = 8  # 10 for most real scenes. 6 or 7 work well for most synthetic scenes
    poisson_depth = lp.poisson_depth if depth is None else depth
    vertices_density_quantile = 0.  # 0.1 for most real scenes. 0. works well for most synthetic scenes
    
    surface_level = 0.3
    #decimation_target = opt.max_num_densify
    decimation_target = opt.mesh_decim_target
            
    # Load the coarse model
    sugar.eval()
    
    # Pruning low opacity gaussians
    with torch.no_grad():
        sugar.drop_low_opacity_points(low_opacity_gaussian_pruning_threshold)

    # Build the triangle soup that will be used for splatting
    sugar.primitive_types = 'diamond'
    sugar.triangle_scale = 2.
    sugar.update_texture_features()
    mesh = sugar.mesh
    
    # Create a mesh renderer
    faces_per_pixel = 10
    max_faces_per_bin = 50_000
    euler_char = 1
    #bin_size = 2 * (decimation_targets[-1] - euler_char)
    bin_size=None

    #print(f'sugar.image_height: {sugar.image_height}, sugar.image_width: {sugar.image_width}') # 1024, 1024
    mesh_raster_settings = RasterizationSettings(
        image_size=(sugar.image_height, sugar.image_width),
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel,
        max_faces_per_bin=max_faces_per_bin,
        bin_size=bin_size
    )
    rasterizer = MeshRasterizer(
            cameras=nerfmodel.training_cameras.p3d_cameras[0], 
            raster_settings=mesh_raster_settings,
    )

    # Compute surface levels point clouds
    n_pts_per_frame = int(n_total_points / len(nerfmodel.training_cameras)) + 1
    sugar.knn_to_track = surface_level_knn_to_track

    surface_levels_outputs = {}
    surface_levels_outputs[surface_level] = {
        'points': torch.zeros(0, 3, device=sugar.device),
        'colors': torch.zeros(0, 3, device=sugar.device),
        'view_directions': torch.zeros(0, 3, device=sugar.device),
        'pix_to_gaussians': torch.zeros(0, dtype=torch.long, device=sugar.device),
        'normals': torch.zeros(0, 3, device=sugar.device),
    }

    with torch.no_grad():
        cameras_to_use = nerfmodel.training_cameras
            
        for cam_idx in range(len(nerfmodel.training_cameras)):
            point_depth = cameras_to_use.p3d_cameras[cam_idx].get_world_to_view_transform().transform_points(sugar.points)[..., 2:].expand(-1, 3)
            
            # Render RGB image with Gaussian splatting
            rgb = sugar.render_image_gaussian_rasterizer_2dgs(
                nerf_cameras=cameras_to_use, 
                camera_indices=cam_idx,
                bg_color = None,
                sh_deg=0,  # nerfmodel.gaussians.active_sh_degree,
                compute_color_in_rasterizer=True,
                compute_covariance_in_rasterizer=True,
                return_2d_radii=False,
                use_same_scale_in_all_directions=False,
            ).clamp(min=0., max=1.).contiguous()
            
            # Compute surface level points for the current frame
            if cam_idx == 0:
                sugar.reset_neighbors(knn_to_track=surface_level_knn_to_track)
            with torch.no_grad():
                frame_surface_level_outputs = sugar.compute_level_surface_points_from_camera_fast_2dgs(
                    cam_idx=cam_idx,
                    rasterizer=rasterizer,
                    surface_levels=[surface_level], 
                    n_surface_points=2*n_pts_per_frame,  # TODO: 2*n_pts_per_frame is safe to avoid empty pixels
                    primitive_types=surface_level_primitive_types, 
                    triangle_scale=surface_level_triangle_scale,
                    splat_mesh=surface_level_splat_mesh,
                    n_points_in_range=surface_level_n_points_in_range,
                    range_size=surface_level_range_size,
                    n_points_per_pass=surface_level_n_points_per_pass,
                    density_factor=1.,
                    return_pixel_idx=True,
                    return_gaussian_idx=True,
                    return_normals=True,
                    compute_flat_normals=flat_surface_level_normals,
                    use_gaussian_depth=use_gaussian_depth_for_surface_levels,)
            
                img_surface_points = frame_surface_level_outputs[surface_level]['intersection_points']
                surface_gaussian_idx = frame_surface_level_outputs[surface_level]['gaussian_idx']
                img_surface_normals = frame_surface_level_outputs[surface_level]['normals']
                
                pixel_idx = frame_surface_level_outputs[surface_level]['pixel_idx']
                img_surface_colors = rgb.view(-1, 3)[pixel_idx]
            
                img_surface_view_directions = torch.nn.functional.normalize(cameras_to_use.p3d_cameras[cam_idx].get_camera_center() - img_surface_points)
                img_surface_pix_to_gaussians = surface_gaussian_idx.view(-1)
                
                idx = torch.randperm(len(img_surface_points), device=sugar.device)[:n_pts_per_frame]
                
                surface_levels_outputs[surface_level]['points'] = torch.cat([surface_levels_outputs[surface_level]['points'], img_surface_points[idx]], dim=0)
                surface_levels_outputs[surface_level]['colors'] = torch.cat([surface_levels_outputs[surface_level]['colors'], img_surface_colors[idx]], dim=0)
                surface_levels_outputs[surface_level]['view_directions'] = torch.cat([surface_levels_outputs[surface_level]['view_directions'], img_surface_view_directions[idx]], dim=0)
                surface_levels_outputs[surface_level]['pix_to_gaussians'] = torch.cat([surface_levels_outputs[surface_level]['pix_to_gaussians'], img_surface_pix_to_gaussians[idx]], dim=0)
                surface_levels_outputs[surface_level]['normals'] = torch.cat([surface_levels_outputs[surface_level]['normals'], img_surface_normals[idx]], dim=0)

    # -----Processing surface level-----
    surface_points = surface_levels_outputs[surface_level]['points']
    surface_colors = surface_levels_outputs[surface_level]['colors']
    surface_normals = surface_levels_outputs[surface_level]['normals']

    fg_bbox_min_tensor = - fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)
    fg_bbox_max_tensor = fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)
    
    # center bbox
    _cameras_spatial_extent, _camera_average_xyz = sugar.get_cameras_spatial_extent(return_average_xyz=True)
    points_idx = torch.arange(len(surface_points))
    with torch.no_grad():
        fg_bbox_min_tensor_c = fg_bbox_min_tensor + _camera_average_xyz
        fg_bbox_max_tensor_c = fg_bbox_max_tensor + _camera_average_xyz
        fg_mask = (surface_points[points_idx] > fg_bbox_min_tensor_c).all(dim=-1) * (surface_points[points_idx] < fg_bbox_max_tensor_c).all(dim=-1)
        n_fg_mask = fg_mask.sum().item()
        if n_fg_mask == 0: # sometimes this happens
            fg_mask = (surface_points[points_idx] > fg_bbox_min_tensor).all(dim=-1) * (surface_points[points_idx] < fg_bbox_max_tensor).all(dim=-1)
    
    # center bbox
    if n_fg_mask > 0:
        bg_mask = ((surface_points[points_idx] - _camera_average_xyz).abs().max(dim=-1)[0]
                    < bg_bbox_factor * _cameras_spatial_extent) * ~fg_mask
    else:
        bg_mask = (surface_points[points_idx].abs().max(dim=-1)[0] < bg_bbox_factor * sugar.get_cameras_spatial_extent()) * ~fg_mask
    fg_points = surface_points[points_idx][fg_mask]
    fg_colors = surface_colors[points_idx][fg_mask]
    fg_normals = surface_normals[points_idx][fg_mask]

    bg_points = surface_points[points_idx][bg_mask]
    bg_colors = surface_colors[points_idx][bg_mask]
    bg_normals = surface_normals[points_idx][bg_mask]

    # ---Compute foreground mesh---
    fg_points_exist = fg_points.shape[0] > 0
    if fg_points_exist:
        fg_pcd = o3d.geometry.PointCloud()
        fg_pcd.points = o3d.utility.Vector3dVector(fg_points.double().cpu().numpy())
        fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors.double().cpu().numpy())
        fg_pcd.normals = o3d.utility.Vector3dVector(fg_normals.double().cpu().numpy())

        # outliers removal
        cl, ind = fg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
        fg_pcd = fg_pcd.select_by_index(ind)

        o3d_fg_mesh, o3d_fg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            fg_pcd, depth=poisson_depth) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

        if vertices_density_quantile > 0.:
            vertices_to_remove = o3d_fg_densities < np.quantile(o3d_fg_densities, vertices_density_quantile)
            o3d_fg_mesh.remove_vertices_by_mask(vertices_to_remove)
    else:
        o3d_fg_mesh = None
    
    # ---Compute background mesh---
    if bg_points.shape[0] > 0:
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(bg_points.double().cpu().numpy())
        bg_pcd.colors = o3d.utility.Vector3dVector(bg_colors.double().cpu().numpy())
        bg_pcd.normals = o3d.utility.Vector3dVector(bg_normals.double().cpu().numpy())

        # outliers removal
        cl, ind = bg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
        bg_pcd = bg_pcd.select_by_index(ind)

        o3d_bg_mesh, o3d_bg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            bg_pcd, depth=poisson_depth) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

        if vertices_density_quantile > 0.:
            vertices_to_remove = o3d_bg_densities < np.quantile(o3d_bg_densities, vertices_density_quantile)
            o3d_bg_mesh.remove_vertices_by_mask(vertices_to_remove)
    else:
        o3d_bg_mesh = None
    
    print("mesh before decimation:", o3d_fg_mesh)

    # ---Decimate and clean meshes---
    # upsample mesh if #face < decimation_target
    """
    try:
        n_tri = len(o3d_fg_mesh.triangles)
    except AttributeError as e:
        print(f'o3d_fg_mesh is None: {o3d_fg_mesh is None}') # True
        print(f'n_fg_mask: {n_fg_mask}, len(surface_points): {len(surface_points)}') # 0, 0
        print(f'fg_bbox_min_tensor: {fg_bbox_min_tensor}, fg_bbox_max_tensor: {fg_bbox_max_tensor}')
        raise SystemExit(1)
    """
    n_tri = len(o3d_fg_mesh.triangles)
   
    iter = 0
    decimated_o3d_fg_mesh = copy.deepcopy(o3d_fg_mesh)
    if decimate:
        while n_tri < decimation_target: # upsample
            decimated_o3d_fg_mesh = decimated_o3d_fg_mesh.subdivide_midpoint()
            n_tri = len(decimated_o3d_fg_mesh.triangles)
            iter += 1
    
    if o3d_fg_mesh is not None:
        #decimated_o3d_fg_mesh = o3d_fg_mesh.simplify_quadric_decimation(decimation_target)
        if decimate:
            decimated_o3d_fg_mesh = decimated_o3d_fg_mesh.simplify_quadric_decimation(decimation_target)
    else:
        decimated_o3d_fg_mesh = None
        
    if o3d_bg_mesh is not None:                  
        if decimate:
            decimated_o3d_bg_mesh = o3d_bg_mesh.simplify_quadric_decimation(decimation_target)
        else:
            decimated_o3d_bg_mesh = o3d_bg_mesh
    else:
        decimated_o3d_bg_mesh = None

    if decimated_o3d_fg_mesh is not None:
        decimated_o3d_fg_mesh.remove_degenerate_triangles()
        decimated_o3d_fg_mesh.remove_duplicated_triangles()
        decimated_o3d_fg_mesh.remove_duplicated_vertices()
        decimated_o3d_fg_mesh.remove_non_manifold_edges()
    
    if decimated_o3d_bg_mesh is not None:
        decimated_o3d_bg_mesh.remove_degenerate_triangles()
        decimated_o3d_bg_mesh.remove_duplicated_triangles()
        decimated_o3d_bg_mesh.remove_duplicated_vertices()
        decimated_o3d_bg_mesh.remove_non_manifold_edges()
    
    if (decimated_o3d_fg_mesh is not None) and (decimated_o3d_bg_mesh is not None):
        decimated_o3d_mesh = decimated_o3d_fg_mesh + decimated_o3d_bg_mesh
    elif decimated_o3d_fg_mesh is not None:
        decimated_o3d_mesh = decimated_o3d_fg_mesh
    elif decimated_o3d_bg_mesh is not None:
        decimated_o3d_mesh = decimated_o3d_bg_mesh
    else:
        raise ValueError("Both foreground and background meshes are empty. Please provide a valid bounding box for the scene.")

    o3d.io.write_triangle_mesh(mesh_path, decimated_o3d_mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
    print(f"Mesh saved at {mesh_path}. #verts: {len(decimated_o3d_mesh.vertices)}, #faces: {len(decimated_o3d_mesh.triangles)}")
    
    gs_model.initialize_from_sugar_mesh(mesh_path)

    return mesh_path


def extract_mesh_alpha_shape(opt, dataset, gaussians, xyz, mesh_path, alpha=0.03, remove_outlier=False):
    # initialize o3d instance
    pcd = o3d.geometry.PointCloud()
    #xyz = gaussians.get_xyz
    n_gs = xyz.shape[0]
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())

    rgb = SH2RGB(gaussians.get_features[:,0].detach().float())
    pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((n_gs,3)))
    
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    # outliers removal
    if remove_outlier:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
        n_remove = n_gs - len(ind)
        if n_remove > 0:
            pcd = pcd.select_by_index(ind)
            print(f"remove {n_remove} outlier points")
    else:
        n_remove = 0
    
    # mesh recon
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    n_v = len(mesh.vertices)

    # save mesh
    o3d.io.write_triangle_mesh(mesh_path, mesh, write_triangle_uvs=True, write_vertex_colors=False, write_vertex_normals=True)
    print(f"mesh saved at {mesh_path}")

    # get new gaussian attributes
    #print(f'len(mesh.vertices): {len(mesh.vertices)}, len(mesh.triangles): {len(mesh.triangles)}')    

def obj_mesh_recon(opt, dataset, gaussians, iteration, mesh_path, opt_with_coarse=False):
    meshify_method = opt.meshify_method
    if meshify_method == 'poisson':
        depth = dataset.poisson_depth
        ext_start_time = datetime.now()
        extract_mesh_upsample(opt, dataset, gaussians, mesh_path)
        print(f"mesh saved at {mesh_path}")
        ext_end_time = datetime.now()
        mesh_ext_time = ext_end_time - ext_start_time
        print("mesh extraction time: ", mesh_ext_time)
    else:
        raise NotImplementedError

    if opt_with_coarse: # get coarse mesh
        coarse_mesh_path = mesh_path.replace('.ply', '_coarse.ply')
        coarse_meshify_method = opt.coarse_meshify_method
        xyz = gaussians.get_xyz
        sampled_xyz, sample_idx = sample_farthest_points(xyz.view(1, -1, 3), K=2048)
        #print(f'sampled_xyz.shape: {sampled_xyz.shape}, sample_idx.shape: {sample_idx.shape}')
        sampled_xyz = sampled_xyz.squeeze(0)
        sample_idx = sample_idx.squeeze(0)

        # run mesh recon from the sampled xyz
        if coarse_meshify_method == 'alpha_shape':
            ext_start_time = datetime.now()
            
            alpha = dataset.alpha_shape
            extract_mesh_alpha_shape(opt, dataset, gaussians, sampled_xyz, coarse_mesh_path, alpha)

            ext_end_time = datetime.now()
            mesh_ext_time = ext_end_time - ext_start_time
            print("coarse mesh extraction time: ", mesh_ext_time)
        else:
            raise NotImplementedError
