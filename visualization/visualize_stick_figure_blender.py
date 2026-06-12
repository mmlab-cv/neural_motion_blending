import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple
from multiprocessing import current_process
from tqdm import tqdm
if current_process().name == 'MainProcess':  # enables multiprocessing 
    import bpy

def save_blender_file(path):
    print(f"{10 * '*'} saved {path}")
    bpy.ops.wm.save_as_mainfile(filepath=path)


def get_rotation_matrix_between_vectors(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    # get a rotation matrix from v0 to v1
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    cos_t = np.dot(v0, v1)
    sin_t = np.linalg.norm(np.cross(v0, v1))

    u = v0
    v = v1 - np.dot(v0, v1) * v0
    v = v / np.linalg.norm(v)
    w = np.cross(v0, v1)
    w = w / np.linalg.norm(w)

    # change of basis matrix
    C = np.array([u, v, w])

    # rotation matrix in new basis
    R_uvw = np.array([[cos_t, -sin_t, 0],
                      [sin_t, cos_t, 0],
                      [0, 0, 1]])
    # full rotation matrix
    R = C.T @ R_uvw @ C
    return R


class CollectionManager:
    @staticmethod
    def create_collection(collection_name):
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)

    @staticmethod
    def add_object_to_collection(object_name, collection_name):
        object_to_add = bpy.data.objects.get(object_name)
        for other_collection in object_to_add.users_collection:
            other_collection.objects.unlink(object_to_add)
        collection = bpy.data.collections[collection_name]
        collection.objects.link(object_to_add)


class CylinderBone:
    def __init__(self, head_ind: int = None, tail_ind: int = None, unq_name: str = None, radius=0.03, material=None) -> None:
        """
        Use name in case of multiple skeletons with the same bones
        """
        self.head_name = Names.get_joint_name(head_ind, unq_name)
        self.tail_name = Names.get_joint_name(tail_ind, unq_name)
        self.bone_name = Names.get_bone_name(head_ind, tail_ind, unq_name)
        self.unq_name = unq_name

        empty = bpy.data.objects.get('empty_axis')

        self.location = None
        self.rotation = None
        self.head_loc = None
        self.tail_loc = None
        self.radius = radius
        

        self.material = bpy.data.materials.get('base_bone') if material is None else material

        bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=self.radius, radius2=self.radius * 0.6, depth=1, 
                                enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        # bpy.ops.object.shade_smooth()
        cyl = bpy.context.object
        cyl.name = self.bone_name
        cyl.parent = empty
        cyl.data.materials.append(self.material)

    def set_head_and_tail_locations_per_frame(self, head_loc: np.ndarray, tail_loc: np.ndarray, frame=0):
        cyl = bpy.data.objects[self.bone_name]
        v_src = np.array([0, 0, 1])  # default cylinder direction
        v_tgt = head_loc - tail_loc
        v_tgt = v_tgt / np.linalg.norm(v_tgt)
        cyl.rotation_mode = 'QUATERNION'
        ix, iy, iz, w = tuple(R.from_matrix(get_rotation_matrix_between_vectors(v_src, v_tgt)).as_quat())
        cyl.rotation_quaternion[0] = w
        cyl.rotation_quaternion[1] = ix
        cyl.rotation_quaternion[2] = iy
        cyl.rotation_quaternion[3] = iz
        cyl.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # length
        z_dim = float(np.linalg.norm(head_loc - tail_loc))
        cyl.scale.z = z_dim
        cyl.keyframe_insert(data_path="scale", frame=frame)

        # location
        self.location = (head_loc + tail_loc) / 2
        cyl.location = self.location
        cyl.keyframe_insert(data_path="location", frame=frame)
        # if self.head_name is not None:

        joint_head = bpy.data.objects[self.head_name]
        joint_tail = bpy.data.objects[self.tail_name]
        joint_head.location = head_loc
        joint_head.keyframe_insert(data_path="location", frame=frame)
        joint_tail.location = tail_loc
        joint_tail.keyframe_insert(data_path="location", frame=frame)


class Names:
    @staticmethod
    def get_joint_name(joint_ind, unq_name):
        return f"j{joint_ind}" if unq_name is None else f"{joint_ind}_{unq_name}"
    
    @staticmethod
    def get_bone_name(head, tail, unq_name):
        return f"cyl_{head}_{tail}" if unq_name is None else f"cyl_{head}_{tail}_{unq_name}"

    @staticmethod
    def get_collection_name(unq_name: str = None):
        return "animal" if unq_name is None else f"Cylinder_Human_{unq_name}"


class StickFigure:
    @staticmethod
    def visualize(joint_locations, bone_list, 
                  unq_name=None, sphere_radius=0.04, cylinder_radius=0.03, joint_materials=None):
        """
        joint locations: [seq_len, n_joints, 3]
        name: use in case of multiple stick figures visualizations
        """
        collection_name = Names.get_collection_name(unq_name)
        CollectionManager.create_collection(collection_name)
        
        empty = bpy.data.objects.new(name='empty_axis', object_data=None)
        empty.empty_display_type = 'PLAIN_AXES'
        bpy.context.scene.collection.objects.link(empty)  # Add the empty to the scene
        CollectionManager.add_object_to_collection('empty_axis', collection_name)
        
        cyl_bone_dict = StickFigure.create_cylinder_bones(collection_name, bone_list, unq_name, cylinder_radius, joint_materials)
        StickFigure.create_joints(collection_name, bone_list, unq_name, sphere_radius)
        n_frames = joint_locations.shape[0]
        for frame in tqdm(range(n_frames), desc=f"Stick figure sequence{f' {unq_name}' if unq_name else ''}"):
            pose = joint_locations[frame]
            StickFigure.apply_pose(pose, frame, cyl_bone_dict, bone_list)

    @staticmethod
    def create_cylinder_bones(collection_name, bone_list, unq_name, radius=0.03, joint_materials=None) -> Dict[Tuple[int, int], CylinderBone]:
        cyl_bones = {}
        for head, tail in bone_list:
            if joint_materials is None:
                material = None 
            elif type(joint_materials) == dict: 
                material = joint_materials[head]
            else:  # single material for all joints case
                material = joint_materials
            cyl_bone = CylinderBone(head, tail, unq_name, radius, material=material)
            cyl_bones[(head, tail)] = cyl_bone
            CollectionManager.add_object_to_collection(cyl_bone.bone_name, collection_name)
        return cyl_bones

    @staticmethod
    def create_joints(collection_name, bone_list: List[Tuple[int, int]], unq_name: str, radius=0.04):
        joints = set()
        material = bpy.data.materials.get('base_joint')
        empty = bpy.data.objects.get('empty_axis')
        for bone in bone_list:
            for j in bone:
                if j not in joints:
                    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, calc_uvs=False, enter_editmode=False, align='WORLD', scale=(1, 1, 1))
                    # bpy.ops.object.shade_smooth()
                    joint_obj = bpy.context.object
                    joint_obj.data.materials.append(material)
                    joint_obj.parent = empty
                    joint_name = Names.get_joint_name(j, unq_name)
                    bpy.context.object.name = joint_name
                    CollectionManager.add_object_to_collection(joint_name, collection_name)
                    joints.add(j)

    @staticmethod
    def apply_pose(pose, frame: int, cyl_bone_dict: Dict[str, CylinderBone], bone_list):
        """
        pose: (n_joints, 3)
        """
        for head_ind, tail_ind in bone_list:
            head_location = pose[head_ind]
            tail_location = pose[tail_ind]
            cyl_bone = cyl_bone_dict[(head_ind, tail_ind)]
            cyl_bone.set_head_and_tail_locations_per_frame(head_location, tail_location, frame=frame)
