import BVH
from Animation import *
from InverseKinematics import animation_from_positions
import numpy as np 
import os 
from os.path import join as pjoin
from Quaternions import Quaternions
import re
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
import random
import math
import statistics
import torch
import bisect
import re 
from data_loaders.truebones.truebones_utils.param_utils import HML_AVG_BONELEN, FOOT_CONTACT_HEIGHT_THRESH, FACE_JOINTS, DATASET_DIR, MAX_PATH_LEN, ANIMATIONS_DIR, MOTION_DIR, NO_BVHS, FOOT_CONTACT_VEL_THRESH, RAW_DATA_DIR, BVHS_DIR
from utils.rotation_conversions import rotation_6d_to_matrix_np

################## Data Generation #####################
""" Computes orientation based on object type face joints (hips and shoulder) and reeturns root quaternion 
rotatation to ensure it faces z+. face_joint_indx is optional for object_types that are not included
in FACE_JOINTS dict"""
def get_root_quat(joints, object_type, face_joint_indx=None):
    if face_joint_indx is None:
        face_joint_indx = FACE_JOINTS[object_type]
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = joints[:, r_hip] - joints[:, l_hip]
    across2 = joints[:, sdr_r] - joints[:, sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
    forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    root_quat = Quaternions.between(forward, target)
    if object_type == "Anaconda":
        root_quat = Quaternions.from_euler(np.array([0, -np.pi/2, 0]), "xyz") * root_quat
    return root_quat
          
""" put skeleton on ground (xz plane) """
def put_on_ground(anim, ground_height=None):
    if ground_height is None:
        t_pos_global_positions = positions_global(anim)
        ground_height = t_pos_global_positions.min(axis=0).min(axis=0)[1]
    new_positions = anim.positions.copy()
    new_positions[:, 0, 1] -= ground_height
    new_offsets = anim.offsets.copy()
    new_offsets[0, 1] -= ground_height
    new_anim = Animation(anim.rotations.copy(), new_positions, anim.orients.copy(), new_offsets, anim.parents.copy())
    return new_anim, ground_height

""" move motion s.t root xz are at origin on first frame"""
def move_xz_to_origin(anim, root_pose_init_xz=None):
    if root_pose_init_xz is None:
        root_pos_init = anim.positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    new_positions = anim.positions.copy()
    new_positions[:, 0] -= root_pose_init_xz
    new_offsets = anim.offsets.copy()
    new_offsets[0] -= root_pose_init_xz
    new_anim = Animation(anim.rotations.copy(), new_positions, anim.orients.copy(), new_offsets, anim.parents.copy())
    return new_anim, root_pose_init_xz

"""" rotate the motion to initially face z+, ground at xz axis (negative y is below ground)"""
def rotate_to_hml_orientation(anim, object_type, face_joints=None):
    global_pos = positions_global(anim)
    qs_rot = get_root_quat(global_pos, object_type, face_joint_indx=face_joints)[0]
    new_rots = anim.rotations.copy()
    new_rots[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_rots[:, 0]
    new_pos = anim.positions.copy()
    new_pos[:, 0] = qs_rot.repeat(new_rots.shape[0], axis=0) * new_pos[:, 0]
    new_anim = Animation(new_rots, new_pos, anim.orients.copy(), anim.offsets.copy(), anim.parents.copy())
    return new_anim

""" scale skeleton s.t longest armature is of length HML_AVG_BONELEN """
def scale(anim, scale_factor=None):
    if scale_factor is None:
        lengths = offset_lengths(anim)
        mean_len = statistics.mean(lengths)
        scale_factor = HML_AVG_BONELEN/mean_len
    new_anim = Animation(anim.rotations.copy(), anim.positions * scale_factor ,anim.orients.copy(), anim.offsets * scale_factor,
                         anim.parents.copy())
    return new_anim, scale_factor

""" get foot contact """
def get_foot_contact(positions, foot_joints_indices, vel_thresh):
    frames_num, joints_num = positions.shape[:2]
    foot_vel_x = (positions[1:,foot_joints_indices ,0] - positions[:-1,foot_joints_indices ,0]) ** 2
    foot_vel_y = (positions[1:, foot_joints_indices, 1] - positions[:-1, foot_joints_indices, 1]) **2
    foot_vel_z = (positions[1:, foot_joints_indices, 2] - positions[:-1, foot_joints_indices, 2]) **2
    total_vel = foot_vel_x + foot_vel_y + foot_vel_z
    foot_contact_vel_map = np.where(np.logical_and(total_vel <= vel_thresh, np.abs(positions[1:, foot_joints_indices,1]) <= FOOT_CONTACT_HEIGHT_THRESH), 1, 0)
    foot_cont = np.zeros((frames_num-1, joints_num))
    foot_cont[:, foot_joints_indices] = foot_contact_vel_map.astype(int)

    return foot_cont

""" get 6d rotations continuous representation"""
def get_6d_rep(qs):
    qs_ = qs.copy()
    return qs_.rotation_matrix(cont6d=True)

"""" process anim object """
def process_anim(anim, object_type, root_pose_init_xz=None, scale_factor=None, ground_height=None, face_joints=None):
    rotated = rotate_to_hml_orientation(anim, object_type, face_joints) 
    centered, root_pose_init_xz_ = move_xz_to_origin(rotated, root_pose_init_xz)
    scaled, scale_factor_ = scale(centered, scale_factor)
    grounded, ground_height_ = put_on_ground(scaled, ground_height)
    return grounded, root_pose_init_xz_, ground_height_, scale_factor_

""" get object_type common characteristics, extracted from Tsode bvh"""
def get_common_features_from_T_pose(t_pose_bvh, object_type, face_joints=None):
    t_pose_anim, t_pos_names, t_pose_frame_time = BVH.load(t_pose_bvh)
    if face_joints:
        if isinstance(face_joints[0], str):
            face_joints = [t_pos_names.index(name) for name in face_joints]
    # first recover global positions, and then create a brand new non-damaged animation, with position consistent to the offsets 
    t_pose_positions = positions_global(t_pose_anim)
    t_pose_anim, _1, _2 = animation_from_positions(positions=t_pose_positions, parents=t_pose_anim.parents, offsets=t_pose_anim.offsets, iterations=150)
    ground_height=None
    if object_type == "Dragon":
        ground_height=0
    scaled, root_pose_init_xz, ground_height, scale_factor = process_anim(t_pose_anim, object_type, ground_height=ground_height, face_joints=face_joints)
    offsets = offsets_from_positions(positions_global(scaled), scaled.parents)[0]
    if object_type in ["Anaconda", "KingCobra"]: # special handel for snakes 
        suspected_foot_indices = [i for i in range(len(t_pos_names))]
    else:
        suspected_foot_indices = [i for i in range(len(t_pos_names)) if 'toe' in t_pos_names[i].lower() or 'foot' in t_pos_names[i].lower() or 
                                  'phalanx' in t_pos_names[i].lower() or 'hoof' in t_pos_names[i].lower() or 'ashi' in t_pos_names[i].lower()]
                # edge cases
        for si in suspected_foot_indices:
            if si in t_pose_anim.parents:
                #check if all childeren also in suspected_foot_indices, otherwise add them 
                children = [i for i in range(len(t_pos_names)) if t_pose_anim.parents[i] == si]
                for c in children:
                    if c not in suspected_foot_indices:
                        suspected_foot_indices.append(c)
    return root_pose_init_xz, scale_factor, ground_height, offsets, suspected_foot_indices, scaled.rotations, t_pos_names, scaled, face_joints

def get_motion_features(ric_positions, rotations, foot_contact, velocity, max_joints):
    # F = Frames# , J = joints# 
    # parents (J,1)
    # positions (F, J, 3)
    # rotations (F, J, 6)
    # foot_contact (F - 1, J, 1)
    # velocity (F - 1, J, 3)
    # offsets (J, 3)
    
    # feature len = 13 (pos, rot, vel, foot)

    frames, joints = ric_positions.shape[0:2]
    if joints > max_joints:
        max_joints = joints
    pos = ric_positions[:-1]  ## (Frames-1, joints, 3)
    rot = rotations[:-1] ## (frames -1, joints, 6)
    vel = velocity ## (Frames - 1, joints , 3)
    foot = foot_contact.reshape(frames - 1, joints , 1) ## (Frames - 1, 1)
    features= np.concatenate([pos, rot, vel, foot], axis=-1) 
    return features, max_joints

'''return positions in root coords system. Meaning, each frame faces Z+, and the root is at [0, root_height, 0]'''
def get_rifke(global_positions, root_rot):
    positions = global_positions.copy()
    '''Local pose'''
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    '''All pose face Z+'''
    positions = np.repeat(root_rot[:, None], positions.shape[1], axis=1) * positions
    return positions

""" compute new rotations for anim that are relative to a natural tpose """
def compute_rots_from_tpos(tpos_quats, dest_quats, parents):
    new_rots = dest_quats.copy()
    new_rots[:, 0] = new_rots[:, 0] * -tpos_quats[:, 0]
    cum_rots = tpos_quats.copy()
    for j, p in enumerate(parents[1:], start=1):
        cum_rots[:, j] = cum_rots[:, p] * tpos_quats[:, j]
        new_rots[:, j] = cum_rots[:, p] * dest_quats[:, j] * -tpos_quats[:, j] * -cum_rots[:, p]
    return new_rots

""" returns policy for extracting kinematic chains from parent array, 
in attempt to divide the skeleton to meaningful kinchains. h_first mean the head joints are at the 
beggining of the parent array"""
def object_policy(obj):
    if obj in ["Mousey_m", "MouseyNoFingers", "Scorpion", "Raptor2"]:
        return "l_first"
    else:
        return "h_first"

""" returns cont6d params, including joints rotations, root rotation and rotational velocity, 
linear velocity and positions. Unlike BVH (and accordingly, Animation object) in which the parent holds the rotagtion of the child joint, 
in our data structure each joints holds it's own rotation (similar to humanML3D data structure and FK model)"""
def get_bvh_cont6d_params(anim, object_type, face_joints=None):
    positions = positions_global(anim)
    if face_joints is None:
        face_joints = FACE_JOINTS[object_type]
    quat_params = anim.rotations
    r_rot = get_root_quat(positions, object_type, face_joints)
    '''Quaternion to continuous 6D'''
    cont_6d_params = get_6d_rep(quat_params)
    cont_6d_params_reordered = np.zeros_like(cont_6d_params)
    for j, p in enumerate(anim.parents[1:], 1):
        cont_6d_params_reordered[:, j] = cont_6d_params[:, p]
    cont_6d_params_reordered[:, 0] = get_6d_rep(r_rot)
    # (seq_len, 4)
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = r_rot[1:] * velocity
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = r_rot[1:] * -r_rot[:-1]
    # (seq_len, joints_num, 4)
    return cont_6d_params_reordered, r_velocity, velocity, r_rot, positions

""" processes animation, and returns a new animation that aligns with humanML3D in terms of orientation and scale"""
def get_hml_aligned_anim(bvh_path, object_type, root_pose_init_xz, scale_factor, ground_height, tpos_rots, offsets, squared_positions_error, face_joints=None, slice_inds=None):
    if not isinstance(bvh_path, Animation):
        raw_anim, names, frame_time = BVH.load(bvh_path)
        if slice_inds:
            raw_anim = raw_anim[slice_inds[0]:slice_inds[1]]
        #print('frame time', frame_time )
        frames_num, joints_num = raw_anim.positions.shape[:2]
        squared_positions_error[bvh_path] = 0 #np.sum((global_pos - new_global_pos) ** 2)/(anim.positions.shape[0]*anim.positions.shape[1])
        #print("positions mismatch error for file: " + bvh_path + " is " + str(squared_positions_error[bvh_path]))

        ## process animation: rotate to correct orientation, center, put on ground and scale
        processed_anim, _xz, _gh, _sf = process_anim(raw_anim, object_type, root_pose_init_xz, scale_factor, ground_height, face_joints=face_joints)
    else:
        names = list()
        processed_anim = bvh_path
        frames_num = len(processed_anim)

    ## create new animation object in which the rotations are w.r.t the actual Tpos
    tpos_rots_correct_shape  = tpos_rots[None, 0].repeat(frames_num, axis = 0)
    rots = compute_rots_from_tpos(tpos_rots_correct_shape, processed_anim.rotations, processed_anim.parents)
    anim_positions = offsets.copy()[None, :].repeat(frames_num, axis = 0)
    anim_positions[:, 0] = processed_anim.positions[:, 0]
    # create animation object which is defined over correct tpos 
    new_anim = Animation(rots, anim_positions  , processed_anim.orients, offsets, processed_anim.parents)

    return new_anim, names  
    
""" get motion feature representation"""
def get_motion(bvh_path, foot_contact_vel_thresh, object_type, max_joints,root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, squared_positions_error, face_joints=None, slice_inds=None):
    try:
        new_anim, names = get_hml_aligned_anim(bvh_path, object_type, root_pose_init_xz, scale_factor, ground_height, tpos_rots, offsets, squared_positions_error, face_joints, slice_inds)
        ## extract features
        # cont_6d_params, r_velocity, velocity, r_rot, global_positions = get_bvh_cont6d_params(new_anim, object_type)
        cont_6d_params, r_velocity, velocity, r_rot, global_positions = get_bvh_cont6d_params(new_anim, object_type, face_joints=face_joints)
        foot_contact = get_foot_contact(global_positions, foot_indices, foot_contact_vel_thresh) 
        '''Get Joint Rotation Invariant Position Represention'''
        # local velocity wrt root coords system as described in get_rifke definition 
        positions = get_rifke(global_positions, r_rot)
        # root_y = positions[:, 0, 1:2]
        # r_velocity = np.arcsin(r_velocity[:, 2:3])
        # l_velocity = velocity[:, [0, 2]]
        local_vel = np.repeat(r_rot[1:, None], global_positions.shape[1], axis=1) * (global_positions[1:] - global_positions[:-1])
        # root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
        features, max_joints = get_motion_features(positions, cont_6d_params, foot_contact, local_vel, max_joints)
        return features, new_anim.parents, max_joints, new_anim
    except Exception as err:
        print(err)
        return None, None, max_joints, None

""" computes mean and std for a list of motions """
def get_mean_std(data):
    if len(data) > 0:
        Mean = data.mean(axis=0) # (Joints, 25)
        Std = data.std(axis=0) # # (Joints, 25)
        Std[0, :3] = Std[0, :3].mean() / 1.0 # all joints except root ric pos
        Std[0, 3:9] = Std[0, 3:9].mean() / 1.0 # all joints except root rotation
        Std[0, 9:12] = Std[0, 9:12].mean() / 1.0 # all joints except root local velocity

        Std[1:, :3] = Std[1:, :3].mean() / 1.0 # all joints except root ric pos
        Std[1:, 3:9] = Std[1:, 3:9].mean() / 1.0 # all joints except root rotation
        Std[1:, 9:12] = Std[1:, 9:12].mean() / 1.0 # all joints except root local velocity
        if len(Std[:, 12][Std[:, 12]!=0]) > 0:
            Std[:, 12][Std[:, 12]!=0] = Std[:, 12][Std[:, 12]!=0].mean() / 1.0 
        Std[:, 12][Std[:, 12]==0] = 1.0 # replace zeros with ones
        
        return Mean, Std
  
""" compures Relations and Distance marices"""
def create_topology_edge_relations(parents, max_path_len = 5): # joint j+1 contains len(j, j+1)
    edge_types = {'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5, 'ts_token_conn': 6}
    n = len(parents)
    topo_rel = np.zeros((n, n))
    edge_rel = np.ones((n, n)) * edge_types['no_relation'] 
    for i in range(n):
        parent = parents[i]
        ee = True
        for j in range(n):
            parent_j = parents[j]
            """Update edge type"""
            edge_type = edge_types['no_relation']
            if i == j: #self
                edge_type = edge_types['self'] 
            elif parent_j == i: #child
                ee=False
                edge_type = edge_types['child']
            elif j == parent: #parent
                edge_type = edge_types['parent'] 
            elif parent_j == parent: #sibling
                edge_type = edge_types['sibling']
            edge_rel[i, j] = edge_type

            """Update path length type"""
            
            if i == j:
                topo_rel[i, j] = 0      
            elif j < i:
                topo_rel[i, j] = topo_rel[j, i]
            elif parent_j == i: # parent-child relation
                topo_rel[i, j] = 1
            else: #any other 
                topo_rel[i, j] = topo_rel[i, parent_j] + 1
        if ee:
            edge_rel[i, i] = edge_types['end_effector']
            
    topo_rel[topo_rel > max_path_len] = max_path_len
    return edge_rel, topo_rel

""" find tpos bvh"""
def find_tpos_path(bvh_files):
    t_pos_path = None
    for f in bvh_files:
        if "tpos" in f.lower():
            t_pos_path = f
            break
    if t_pos_path is not None:
        bvh_files.remove(t_pos_path)
    else: #choose some other motion to be treated as tpos 
        for f in bvh_files:
            fnam = os.path.basename(f)
            if fnam.lower().startswith('idle') or fnam.lower().startswith('__idle'):
                t_pos_path = f
                break
    if t_pos_path is None:
        t_pos_path = bvh_files[0]
    return t_pos_path
     
""" creates processed tensors for all the files of a given object. Returens statistics and the object condition,
which includes tpos, relation/distances matrices, offsets, parents, joints names, kinematic chains, mean and std"""    
def process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir = DATASET_DIR, face_joints=None, bvhs_dir=None, t_pos_path=None):
    object_cond = dict()
    if bvhs_dir is None:
        bvhs_dir = pjoin(RAW_DATA_DIR, object_type)
    bvh_files = [pjoin(bvhs_dir, f) for f in os.listdir(bvhs_dir) if f.lower().endswith('.bvh')]     
    if len(bvh_files) == 0:
        return files_counter, frames_counter, max_joints
    ## get t-pos bvh
    if t_pos_path is None or t_pos_path == '':
        t_pos_path = find_tpos_path(bvh_files)
    else: 
        # removes tpos bvh fron bvh_files, as it represents a static motion and should be used only for
        # extracting common characteristics. If this is not the case, disable this part
        bvh_files.remove(t_pos_path)
        
    root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, names, tpos_anim, face_joints = get_common_features_from_T_pose(t_pos_path, object_type, face_joints=face_joints)
    t_pos_motion, parents, max_joints, new_anim = get_motion(tpos_anim, FOOT_CONTACT_VEL_THRESH, object_type, max_joints, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, squared_positions_error, face_joints=face_joints)
    object_cond['tpos_first_frame'] = t_pos_motion[0]
    # alignment params — stored so BVH→features conversion at eval time can use
    # exactly the same values as dataset build time, without re-running T-pose extraction
    object_cond['root_pose_init_xz'] = root_pose_init_xz
    object_cond['scale_factor'] = scale_factor
    object_cond['ground_height'] = ground_height
    object_cond['tpos_rots'] = tpos_rots
    object_cond['foot_indices'] = foot_indices
    object_cond['face_joints'] = face_joints
    # create topology conditions
    joint_relations, joints_graph_dist = create_topology_edge_relations(tpos_anim.parents, max_path_len = MAX_PATH_LEN)
    object_cond['joint_relations'] = joint_relations
    object_cond['joints_graph_dist'] = joints_graph_dist
    object_cond['object_type'] = object_type
    object_cond['parents'] = parents
    object_cond['offsets'] = offsets
    object_cond['joints_names'] = names
    kinematic_chains = parents2kinchains(parents, object_policy(object_type))
    object_cond['kinematic_chains'] = kinematic_chains
    all_tensors = list()
    
    for f in bvh_files:
        print("processing file: " + f)
        raw_anim, names, frame_time = BVH.load(f)
        anim_len = len(raw_anim)
        begin = 0
        slice_ind = anim_len
        while begin < anim_len:
            if anim_len - begin > 240:
                slice_ind = begin + 200
            else:
                slice_ind = anim_len
            motion, parents, max_joints, new_anim = get_motion(f, FOOT_CONTACT_VEL_THRESH, object_type, max_joints, root_pose_init_xz, scale_factor, ground_height, offsets, foot_indices, tpos_rots, squared_positions_error, slice_inds=[begin, slice_ind], face_joints=face_joints)
            begin = slice_ind
            if motion is not None:
                _, file_name = os.path.split(f)
                action = os.path.splitext(file_name)[0]
                all_tensors.append(motion)
                files_counter += 1
                frames_counter += motion.shape[0]
                name = object_type + "_" + action + "_" + str(files_counter)
                np.save(pjoin(save_dir, MOTION_DIR, name + '.npy'), motion)
                BVH.save(pjoin(save_dir, BVHS_DIR, name+".bvh"), new_anim, names)
                # create mp4 from rotations (sanity check)
                positions = recover_from_bvh_ric_np(motion)
                fc = [[j for j in range(len(parents)) if motion[f, j , 12] != 0] for f in range(motion.shape[0])]
                plot_general_skeleton_3d_motion(pjoin(save_dir, ANIMATIONS_DIR, name+"_from_ric.mp4"), parents, positions, dataset="truebones", title="", fps=20, face_joints=face_joints if face_joints is not None else FACE_JOINTS[object_type], fc = fc)
        
            else:
                print(f'failed to process file: {f}, slice {begin}:{slice_ind}')
    all_tensors = np.concatenate(all_tensors, axis=0)
    mean, std = get_mean_std(all_tensors)
    object_cond["mean"] = mean
    object_cond["std"] = std

    return files_counter, frames_counter, max_joints, object_cond

""" create dataset """
def create_data_samples():
    ## prepare
    os.makedirs(pjoin(DATASET_DIR, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(DATASET_DIR, ANIMATIONS_DIR), exist_ok=True)
    os.makedirs(pjoin(DATASET_DIR, BVHS_DIR), exist_ok=True)
    
    ## process
    objects = [obj for obj in FACE_JOINTS.keys() if FACE_JOINTS[obj] != []]
    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    
    for object_type in objects:
        if object_type in NO_BVHS:
            continue
        cur_counter = files_counter
        files_counter, frames_counter, max_joints, object_cond = process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error)
        cond[object_type] = object_cond
        objects_counter[object_type] = files_counter - cur_counter 

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(DATASET_DIR, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(DATASET_DIR, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()
    
    np.save(pjoin(DATASET_DIR, "cond.npy"), cond)
##################################################################

############ Recover animation from motion features ##############
def recover_root_quat_and_pos_np(data):
    # root_feature_vector.shape = (frames, angular_vel || linear_xz_vel || root_height || zero pad)
    r_rot_quat = Quaternions.from_transforms(rotation_6d_to_matrix_np(data[:, 3:9]))

    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
    '''Add Y-axis rotation to root position'''
    r_pos = -r_rot_quat * r_pos

    r_pos = np.cumsum(r_pos, axis = -2)
    r_pos[...,1] = data[..., 1]
    return r_rot_quat, r_pos

""" recover quaternions and positions from features for numpy only"""
def recover_root_quat_and_pos(data):
    # root_feature_vector.shape = (frames, angular_vel || linear_xz_vel || root_height || zero pad)
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    r_rot_quat = Quaternions(r_rot_quat)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = -r_rot_quat * r_pos

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

""" recover xyz positions from ric (root relative positions) torch """
def recover_from_bvh_ric_np(data):
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[..., 0, :])
    positions = data[..., 1:, :3]
    positions = np.repeat(-r_rot_quat[..., None, :], positions.shape[-2], axis=-2) * positions
    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    '''Concate root and joints'''
    positions = np.concatenate([r_pos[..., np.newaxis, :], positions], axis=-2)
    return positions

""" recover xyz positions from rot (root relative positions) torch """
def recover_from_bvh_rot_np(data, parents, offsets):
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[:,0])
    r_rot_cont6d = get_6d_rep(r_rot_quat)
    start_indx = 3
    end_indx = 9
    cont6d_params = data[..., 1:, start_indx:end_indx]
    cont6d_params = np.concatenate([r_rot_cont6d[:, None, :], cont6d_params], axis=-2)
    cont6d_params_hml_order = rotation_6d_to_matrix_np(cont6d_params)
    cont6d_params = np.eye(3)[None, None].repeat(cont6d_params.shape[0], axis=0).repeat(cont6d_params.shape[1], axis=1)
    for j, p in enumerate(parents[1:], 1):
        cont6d_params[:, p] = cont6d_params_hml_order[:, j]
    rotations = Quaternions.from_transforms(cont6d_params)
    rotations[:, 0] = -r_rot_quat * rotations[:, 0]
    positions = offsets[None].repeat(data.shape[0], axis=0)
    positions[:, 0] = r_pos
    anim = Animation(rotations=rotations, positions=positions, parents=parents, offsets=offsets, orients=Quaternions.id(0))
    
    return positions_global(anim), anim

################################################################

################ Parents to kinematic chains ###################
def reverse_insort(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    a.insert(lo, x)

def parents2kinchains(parents, policy = 'h_first'):
    chains = list()
    children_dict = {i:[] for i in range(len(parents))}
    for j,p in enumerate(parents[1: ], start=1):
        if policy == 'h_first':
            reverse_insort(children_dict[p], j)
        else:
            bisect.insort(children_dict[p], j)
    recursion_kinchains([], 0, children_dict, chains, policy)
    return chains

def recursion_kinchains(chain, j, children_dict, chains, policy):
    children = children_dict[j]
    if len(children) == 0: #ee
        chain.append(j)
        chains.append(chain) 
    elif len(children) == 1:
        chain.append(j)
        recursion_kinchains(chain, children[0], children_dict, chains, policy)
    else:
        chain.append(j)
        if policy == 'h_first':
            main_child = max(children)
        else:
            main_child = min(children)
        for child in children:
            if child == main_child:
                recursion_kinchains(chain, child, children_dict, chains, policy)
            else:
                recursion_kinchains([j], child, children_dict, chains, policy)  
      
################################################################

####################### Augmentations ##########################
"""
def remove_joints_augmentation(data, removal_rate, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    ee = [chain[-1] for chain in kinematic_chains]
    possible_feet = np.unique(np.where(motion[..., -1] > 0)[1])
    if object_type in ['KingCobra', 'Anaconda']:
        possible_feet=[]
    removal_options = [j for j in ee if j not in possible_feet]
    # removal_rate = min(1.0, (removal_rate*len(parents)) / len(removal_options))
    remove_joints = sorted(random.sample(removal_options, math.floor(len(removal_options) * removal_rate)), reverse=True)
    motion = np.delete(motion, remove_joints, axis=1)
    new_ee = [parents[j] for j in remove_joints if np.count_nonzero(parents == parents[j]) == 1]
    for el in new_ee:
        joints_relations[el, el] = 5    
    parents = np.delete(parents, remove_joints, axis=0)
    joints_relations = np.delete(np.delete(joints_relations, remove_joints, axis=0), remove_joints, axis=1)
        
    for rj in remove_joints:
        parents[parents > rj] -= 1
    joints_graph_dist = np.delete(np.delete(joints_graph_dist, remove_joints, axis=0), remove_joints, axis=1)
    tpos_first_frame = np.delete(tpos_first_frame, remove_joints, axis=0)
    offsets = np.delete(offsets, remove_joints, axis=0)
    joints_names_embs = np.delete(joints_names_embs, remove_joints, axis=0)
    mean = np.delete(mean, remove_joints, axis=0)
    std = np.delete(std, remove_joints, axis=0)
    object_type = f'{object_type}__remove{remove_joints}'
    return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std
"""
def remove_joints_augmentation(data, removal_rate, mean, std):
    # 1. Unpack data
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = \
        data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], \
        data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']

    # 2. Identify removal candidates (End Effectors that aren't feet)
    ee = [chain[-1] for chain in kinematic_chains]
    possible_feet = np.unique(np.where(motion[..., -1] > 0)[1])
    if object_type in ['KingCobra', 'Anaconda']:
        possible_feet = []
    removal_options = [j for j in ee if j not in possible_feet]
    
    # 3. Select joints to remove
    remove_joints = sorted(random.sample(removal_options, math.floor(len(removal_options) * removal_rate)), reverse=True)

    # 4. Handle End Effector logic for remaining parents
    new_ee = [parents[j] for j in remove_joints if np.count_nonzero(parents == parents[j]) == 1]
    for el in new_ee:
        joints_relations[el, el] = 5

    # 5. Physical deletion from arrays
    motion = np.delete(motion, remove_joints, axis=1)
    parents = np.delete(parents, remove_joints, axis=0)
    joints_relations = np.delete(np.delete(joints_relations, remove_joints, axis=0), remove_joints, axis=1)
    joints_graph_dist = np.delete(np.delete(joints_graph_dist, remove_joints, axis=0), remove_joints, axis=1)
    tpos_first_frame = np.delete(tpos_first_frame, remove_joints, axis=0)
    offsets = np.delete(offsets, remove_joints, axis=0)
    joints_names_embs = np.delete(joints_names_embs, remove_joints, axis=0)
    mean = np.delete(mean, remove_joints, axis=0)
    std = np.delete(std, remove_joints, axis=0)

    # 6. Update Kinematic Chains
    updated_kinematic_chains = []
    for chain in kinematic_chains:
        new_chain = [j for j in chain if j not in remove_joints]
        shifted_chain = []
        for j in new_chain:
            shift = sum(1 for rj in remove_joints if rj < j)
            shifted_chain.append(j - shift)
        if shifted_chain:
            updated_kinematic_chains.append(shifted_chain)
    kinematic_chains = updated_kinematic_chains

    # 7. Final index shifting for the parents array
    for rj in remove_joints:
        parents[parents > rj] -= 1
    object_type = f'{object_type}__remove{remove_joints}'
    return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, \
           tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std

"""
def add_joint_augmentation(data, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    n_joints = motion.shape[1]
    n_frames = motion.shape[0]
    # added joint mut follow:
    # j has exactly 1 child 
    # j parent is not the root joint
    # j is not the root joint
    possible_joints_to_add = [j for j in range(1, n_joints) if np.count_nonzero(joints_relations[j] == 2) == 1 and joints_relations[j,0] != 1]
    if len(possible_joints_to_add) == 0:
        return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std
    add_j = random.choice(possible_joints_to_add)
    # motion features
    j_feats = motion[:, add_j].copy()
    p_feats = motion[:, parents[add_j]]
    new_feats = ((j_feats + p_feats)/2).copy()
    new_feats[..., 3:9] = j_feats[..., 3:9].copy() # rotations
    new_feats[..., 12] = j_feats[..., 12].copy() # feet 
    j_feats[..., 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])[None].repeat(n_frames, axis=0)
    
    # tpos features
    tpos_j_feats = tpos_first_frame[add_j].copy()
    tpos_p_feats = tpos_first_frame[parents[add_j]]
    tpos_new_feats = ((tpos_j_feats + tpos_p_feats)/2)
    tpos_new_feats[3:9] = tpos_j_feats[3:9].copy() # rotations
    tpos_new_feats[12] = tpos_j_feats[12] # feet 
    tpos_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # mean features
    mean_j_feats = mean[add_j].copy()
    mean_p_feats = mean[parents[add_j]]
    mean_new_feats = ((mean_j_feats + mean_p_feats)/2).copy()
    mean_new_feats[3:9] = mean_j_feats[3:9].copy() # rotations
    mean_new_feats[12] = mean_j_feats[12] # feet 
    mean_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # std features
    std_new_feats = std[add_j].copy()
    
    # joints names embs features 
    emb_j_feats = joints_names_embs[add_j]
    emb_p_feats = joints_names_embs[parents[add_j]]
    emb_new_feats = (emb_j_feats + emb_p_feats)/2
    
    # apply augmentation
    #motion
    augmented = np.concatenate([motion[:, :add_j], new_feats[:, None], j_feats[:, None], motion[:, add_j+1:]], axis=1).copy()
    #tpos_first_frame
    tpos_first_frame_augmented = np.vstack([tpos_first_frame[:add_j], tpos_new_feats[None], tpos_j_feats[None], tpos_first_frame[add_j+1:]]).copy()
    #mean TODO: AUGMENT LIKE MOTION AND TPOS 
    mean_augmented = np.vstack([mean[:add_j], mean_new_feats[None], mean_j_feats[None], mean[add_j+1:]]).copy()
    #std TODO: AUGMENT LIKE MOTION AND TPOS 
    std_augmented = np.vstack([std[:add_j], std_new_feats[None], std[add_j:]]).copy()
    #joints_names_embs
    joints_names_embs_augmented = np.vstack([joints_names_embs[:add_j], emb_new_feats[None], joints_names_embs[add_j:]]).copy()
    # parents 
    augmented_parents = parents.copy()
    augmented_parents[augmented_parents >= add_j] += 1
    augmented_parents = augmented_parents.tolist()
    augmented_parents = np.array(augmented_parents[:add_j] + [add_j] + augmented_parents[add_j:])

    # topology conditions 
    relations, graph_dist = create_topology_edge_relations(augmented_parents.tolist(), max_path_len = MAX_PATH_LEN)
    
    # all others 
    offsets = np.vstack([offsets[:add_j], offsets[add_j]/2, offsets[add_j]/2, offsets[add_j+1:]])
    object_type = f'{object_type}__add{add_j}'
    return augmented, m_length, object_type, augmented_parents, graph_dist, relations, tpos_first_frame_augmented, offsets, joints_names_embs_augmented, kinematic_chains, mean_augmented, std_augmented
"""
def add_joint_augmentation(data, mean, std):
    motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains = data['motion'], data['length'], data['object_type'], data['parents'], data['joints_graph_dist'], data['joints_relations'], data['tpos_first_frame'], data['offsets'], data['joints_names_embs'], data['kinematic_chains']
    n_joints = motion.shape[1]
    n_frames = motion.shape[0]
    
    # added joint mut follow:
    # j has exactly 1 child 
    # j parent is not the root joint
    # j is not the root joint
    possible_joints_to_add = [j for j in range(1, n_joints) if np.count_nonzero(joints_relations[j] == 2) == 1 and joints_relations[j,0] != 1]
    
    if len(possible_joints_to_add) == 0:
        return motion, m_length, object_type, parents, joints_graph_dist, joints_relations, tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std
    
    add_j = random.choice(possible_joints_to_add)
    
    # motion features
    j_feats = motion[:, add_j].copy()
    p_feats = motion[:, parents[add_j]]
    new_feats = ((j_feats + p_feats)/2).copy()
    new_feats[..., 3:9] = j_feats[..., 3:9].copy() # rotations
    new_feats[..., 12] = j_feats[..., 12].copy() # feet 
    j_feats[..., 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])[None].repeat(n_frames, axis=0)
    
    # tpos features
    tpos_j_feats = tpos_first_frame[add_j].copy()
    tpos_p_feats = tpos_first_frame[parents[add_j]]
    tpos_new_feats = ((tpos_j_feats + tpos_p_feats)/2)
    tpos_new_feats[3:9] = tpos_j_feats[3:9].copy() # rotations
    tpos_new_feats[12] = tpos_j_feats[12] # feet 
    tpos_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # mean features
    mean_j_feats = mean[add_j].copy()
    mean_p_feats = mean[parents[add_j]]
    mean_new_feats = ((mean_j_feats + mean_p_feats)/2).copy()
    mean_new_feats[3:9] = mean_j_feats[3:9].copy() # rotations
    mean_new_feats[12] = mean_j_feats[12] # feet 
    mean_j_feats[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # std features
    std_new_feats = std[add_j].copy()
    
    # joints names embs features 
    emb_j_feats = joints_names_embs[add_j]
    emb_p_feats = joints_names_embs[parents[add_j]]
    emb_new_feats = (emb_j_feats + emb_p_feats)/2
    
    # apply augmentation
    augmented = np.concatenate([motion[:, :add_j], new_feats[:, None], j_feats[:, None], motion[:, add_j+1:]], axis=1).copy()
    tpos_first_frame_augmented = np.vstack([tpos_first_frame[:add_j], tpos_new_feats[None], tpos_j_feats[None], tpos_first_frame[add_j+1:]]).copy()
    mean_augmented = np.vstack([mean[:add_j], mean_new_feats[None], mean_j_feats[None], mean[add_j+1:]]).copy()
    std_augmented = np.vstack([std[:add_j], std_new_feats[None], std[add_j:]]).copy()
    joints_names_embs_augmented = np.vstack([joints_names_embs[:add_j], emb_new_feats[None], joints_names_embs[add_j:]]).copy()
    
    # parents 
    augmented_parents = parents.copy()
    augmented_parents[augmented_parents >= add_j] += 1
    augmented_parents = augmented_parents.tolist()
    augmented_parents = np.array(augmented_parents[:add_j] + [add_j] + augmented_parents[add_j:])

    # topology conditions 
    relations, graph_dist = create_topology_edge_relations(augmented_parents.tolist(), max_path_len = MAX_PATH_LEN)
    
    # all others 
    offsets = np.vstack([offsets[:add_j], offsets[add_j]/2, offsets[add_j]/2, offsets[add_j+1:]])
    object_type = f'{object_type}__add{add_j}'

    #  Update Kinematic Chains
    kinematic_chains_updated = []
    for chain in kinematic_chains:
        new_chain = []
        for j in chain:
            shifted_j = j + 1 if j >= add_j else j            
            if j == add_j:
                new_chain.append(add_j)
            new_chain.append(shifted_j)
        kinematic_chains_updated.append(new_chain)

    return augmented, m_length, object_type, augmented_parents, graph_dist, relations, tpos_first_frame_augmented, offsets, joints_names_embs_augmented, kinematic_chains_updated, mean_augmented, std_augmented

########################### Tests ##############################
def process_single_object_type(object_type, save_dir):
    ## prepare
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, ANIMATIONS_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)
    
    ## process
    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    if object_type in NO_BVHS:
        print(f"No bvh files exist for object_type {object_type}")
        exit(1)
    cur_counter = files_counter
    files_counter, frames_counter, max_joints, object_cond = process_object(object_type, files_counter, frames_counter, max_joints, squared_positions_error, save_dir=save_dir)
    cond[object_type] = object_cond
    objects_counter[object_type] = files_counter - cur_counter 

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(save_dir, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(save_dir, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()
    
    np.save(pjoin(save_dir, "cond.npy"), cond)
    
    
def process_skeleton(object_name, bvh_dir, face_joints, save_dir, tpos_bvh=None):
    ## prepare
    os.makedirs(pjoin(save_dir, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, ANIMATIONS_DIR), exist_ok=True)
    os.makedirs(pjoin(save_dir, BVHS_DIR), exist_ok=True)
    
    ## process
    files_counter = 0
    frames_counter = 0
    max_joints = 23
    objects_counter = dict()
    squared_positions_error = dict()
    cond = dict()
    cur_counter = files_counter
    files_counter, frames_counter, max_joints, object_cond = process_object(object_name, files_counter, frames_counter, max_joints, squared_positions_error, save_dir=save_dir, bvhs_dir=bvh_dir, face_joints=face_joints, t_pos_path=tpos_bvh)
    cond[object_name] = object_cond
    objects_counter[object_name] = files_counter - cur_counter 

    print('Total clips: %d, Frames: %d, Duration: %fm' %(files_counter, frames_counter, frames_counter / 12.5 / 60))
    print('max joints: %d' %(max_joints))
    text_file = open(pjoin(save_dir, 'metadata.txt'), "w")
    n = text_file.write('max joints: %d\n' %(max_joints))
    n = text_file.write('total frames: %d\n' %(frames_counter))
    n = text_file.write('duration: %d\n' %(frames_counter / 12.5 / 60))
    n = text_file.write('~~~~ objects_counts - Total: %d ~~~~\n' %(files_counter) )
    for obj in objects_counter:
        text_file.write('%s: %d\n' %(obj, objects_counter[obj]))
    text_file.close()

    error_file = open(pjoin(save_dir, 'positions_error_rate.txt'), "w")
    n = error_file.write('Position squared error per bvh file:')
    for f in squared_positions_error.keys():
        error_file.write('%s: %f\n' %(f, squared_positions_error[f]))
    error_file.close()
    
    np.save(pjoin(save_dir, "cond.npy"), cond)
################################################################