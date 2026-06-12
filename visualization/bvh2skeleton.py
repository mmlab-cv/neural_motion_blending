# Run with Blender!
import bpy
import sys
sys.path.append('.')
import numpy as np
import os
import BVH
from visualization.visualize_stick_figure_blender import StickFigure, save_blender_file
import Animation
from utils.parser_util import render_parser
from mathutils import Vector, Euler

CONNECTED_TO_GROUND = ["Bear", "Camel", "Hippopotamus", "Horse", "Pirrana", "Pteranodon", "Raptor3", "Rat", "SabreToothTiger", "Scorpion-2", "Spider", "Trex", "Tukan", "Pirrana"]
BAD_CHARS = ["Bird", "Giantbee", "Parrot", "Parrot2", "Pigeon"]

SUBSET_TO_SCALE = {"bipeds": 0.7, "flying": 0.5, "millipeds_snakes": 0.507, "quadropeds": 0.507}
SUBSET_TO_LOCATION = {"bipeds": Vector((-1.58, 0.1379864662885666, 0.04131503775715828)) , "flying": Vector((-1.58, 0.137, -0.2)), "millipeds_snakes": Vector((-2, 0.1379864662885666, 0.08)), "quadropeds": Vector((-1.58, 0.1379864662885666, 0.04131503775715828))}

if __name__ == "__main__":
    args = render_parser()
    if args.bvh_path.endswith('.bvh'):
        all_bvh_path = [args.bvh_path]
    elif os.path.isdir(args.bvh_path):
        all_bvh_path = [os.path.join(args.bvh_path, f) for f in os.listdir(args.bvh_path) if f.endswith('.bvh')]
    else:
        raise ValueError()
    
    for b_path in all_bvh_path:

        bpy.ops.wm.read_homefile(use_empty=True)
        bvh_character = os.path.basename(b_path).split('__')[0]
        print(f'Character [{bvh_character}]')

        if bvh_character in CONNECTED_TO_GROUND + BAD_CHARS:
            print('Skipping characters connected to the ground.')
            continue

        os.makedirs(args.save_dir, exist_ok=True)
        out_name = ''.join(b_path.split(os.sep)[-2:])

        blend_path = os.path.join(args.save_dir, out_name.replace('.bvh', '.blend'))

        motion, joint_names, dt = BVH.load(b_path)
        bone_starts = np.arange(len(motion.parents))
        bone_ends = motion.parents
        bones = [(s,e) for s,e in zip(bone_starts, bone_ends)][1:]
        poses = Animation.positions_global(motion) * args.scale
        bpy.context.scene.frame_end = poses.shape[0]
        StickFigure.visualize(joint_locations=poses, bone_list=bones, cylinder_radius=args.cylinder_radius, sphere_radius=args.sphere_radius, joint_materials=None)
        
        empty = bpy.data.objects.get('empty_axis')
        scale = args.scale
        empty.scale = Vector((scale, scale, scale))
        empty.rotation_euler = Euler((1.5707963705062866, 0.0, 2.093), 'XYZ')
        empty.location = SUBSET_TO_LOCATION[args.subset] 

        save_blender_file(blend_path)
        print(blend_path)
