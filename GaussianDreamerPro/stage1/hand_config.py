import os
import os.path as osp
import sys

class Config:
    
    ## shape
    triplane_shape_3d = (2, 2, 2)
    triplane_face_shape_3d = (0.3, 0.3, 0.3)
    triplane_shape = (32, 128, 128)
    
    subdivide_num = 4

    mano_rhand = 1

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..', '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'hand_model')
    vis_dir = osp.join(output_dir, 'hand_vis')
    log_dir = osp.join(output_dir, 'hand_log')
    result_dir = osp.join(output_dir, 'hand_result')
    human_model_path = osp.join(root_dir, 'load', 'human_model_files')

    def set_args(self, subject_id, fit_pose_to_test=False, continue_train=False):
        self.subject_id = subject_id
        self.fit_pose_to_test = fit_pose_to_test
        self.continue_train = continue_train
        if self.fit_pose_to_test:
            self.smplx_param_lr = 1e-3
            self.mano_param_lr = 1e-3
            self.model_dir = osp.join(self.model_dir, subject_id + '_fit_pose_to_test')
            self.result_dir = osp.join(self.result_dir, subject_id + '_fit_pose_to_test')
        else:
            self.smplx_param_lr = 1e-4
            self.mano_param_lr = 1e-4
            self.model_dir = osp.join(self.model_dir, subject_id)
            self.result_dir = osp.join(self.result_dir, subject_id)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
    
cfg = Config()
