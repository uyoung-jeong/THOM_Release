import pickle


class ObjectTHOMModel:
    def __init__(self, obj_dict):
        self.obj_dict = obj_dict
        
        self.obj_pcs = obj_dict["obj_pcs"]
        self.obj_pc_normals = obj_dict["obj_pc_normals"]
        self.point_sets = obj_dict["point_sets"]
        self.obj_path = obj_dict["obj_path"]
        self.obj_pc_top = None
        self.verts_raw = obj_dict['verts_raw']
        self.scale_factor = obj_dict['scale_factor']

    def __call__(self, **kwargs):
        point_set = self.point_sets.copy()
        obj_pc = self.obj_pcs.copy()
        obj_pc_normal = self.obj_pc_normals.copy()
        obj_path = self.obj_path
        if self.obj_pc_top is not None:
            obj_pc_top = self.obj_pc_top.copy()
            return point_set, obj_pc, obj_pc_normal, obj_path, obj_pc_top
        else:
            return point_set, obj_pc, obj_pc_normal, obj_path

def build_object_thom_model(obj_dict):
    object_model = ObjectTHOMModel(obj_dict)
    return object_model