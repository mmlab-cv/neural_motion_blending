import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import os 
from textwrap import wrap
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import clips_array
from pathlib import Path


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new

def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=None)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue
    
    n_frames = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        # sometimes index is equal to n_frames/fps due to floating point issues. in such case, we duplicate the last frame
        index = min(n_frames-1, int(index*fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)

        plt.axis('off')
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


        return mplfig_to_npimage(fig)

    ani = VideoClip(update)
    
    plt.close()
    return ani

def get_general_skeleton_3d_motion(parents, joints, title, dataset, figsize=(7, 7), fps=120, radius=5, face_joints = [], fc = None, title_wrap=20):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, title_wrap))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=None)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    data = joints.copy().reshape(len(joints), -1, 3)
    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    # elif dataset in ['truebones']: 
    #     data *= 0.2
    elif dataset in ['humanml', 'truebones', 'humanml_mat']:
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    n_frames = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]


    def update(index):
        index = min(n_frames-1, int(index*fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        for joint, parent in enumerate(parents[1:], start=1):
            ax.plot3D(data[index, [joint, parent], 0], data[index, [joint, parent], 1], data[index, [joint, parent], 2], color='red', solid_capstyle='round')
            if joint in face_joints:
                ax.scatter(data[index, joint, 0], data[index, joint, 1], data[index,joint, 2], color='blue', marker='o')
            if fc is not None and joint in fc[index]:
                ax.scatter(data[index, joint, 0], data[index, joint, 1], data[index,joint, 2], color='green', marker='o')
        
        plt.axis('off')
        ax.set_axis_off()
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        	
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)

    plt.close()
    return ani 

def plot_general_skeleton_correspondance(parents, joint2color, n_colors, joints, title, dataset, figsize=(7, 7), fps=120, radius=5):
    matplotlib.use('Agg')
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=None)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    data = joints.copy().reshape(len(joints), -1, 3)
    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    # elif dataset in ['truebones']: 
    #     data *= 0.2
    elif dataset in ['humanml', 'truebones', 'humanml_mat']:
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    n_frames = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]


    def update(index):
        index = min(n_frames-1, int(index*fps))
        ax.clear()
        ax.view_init(elev=90, azim=-90)
        ax.dist = 7.5
        # plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
        #              MAXS[2] - trajec[index, 1])
        for joint, parent in enumerate(parents[1:], start=1):
            color = colors[joint2color[index][joint]]
            ax.plot3D(data[index, [joint, parent], 0], data[index, [joint, parent], 1], data[index, [joint, parent], 2], color=color, solid_capstyle='round', linewidth=3.0)

        
        plt.axis('off')
        ax.set_axis_off()
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        	
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)

    plt.close()
    return ani 

def plot_skeleton_spatial_map(parents, joint2color, color_list, joints, title, dataset, figsize=(7, 7), fps=120, radius=5):
    matplotlib.use('Agg')
    
    # Preserve original title wrapping (20 chars)
    title = '\n'.join(wrap(title, 20))

    # Preserve original data reshaping and scaling
    data = joints.copy().reshape(len(joints), -1, 3)
    if dataset == 'kit':
        data *= 0.003
    elif dataset in ['humanml', 'truebones', 'humanml_mat']:
        data *= 1.3
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    
    # Original init() logic
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(False)

    init()
    
    # Original Coordinate centering logic
    MINS = data.min(axis=0).min(axis=0)
    n_frames = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        # Preserve original index calculation: min(n_frames-1, int(index*fps))
        # Note: MoviePy passes index as a float time or sequence index depending on VideoClip config
        frame_idx = min(n_frames-1, int(index*fps))
        
        ax.clear()
        # Preserve original view and distance
        ax.view_init(elev=90, azim=-90)
        ax.dist = 7.5
        
        # Preserve original bone plotting logic
        for joint, parent in enumerate(parents[1:], start=1):
            # Semantic Color Lookup
            color_idx = joint2color[frame_idx][joint]
            
            if color_idx >= len(color_list):
                color = (0.5, 0.5, 0.5, 1.0) # Fallback gray
            else:
                color = color_list[color_idx]

            # Preserve ax.plot3D style (linewidth 3.0, solid_capstyle)
            ax.plot3D(
                data[frame_idx, [joint, parent], 0], 
                data[frame_idx, [joint, parent], 1], 
                data[frame_idx, [joint, parent], 2], 
                color=color, 
                solid_capstyle='round', 
                linewidth=3.0
            )

        # Preserve original UI cleaning
        plt.axis('off')
        ax.set_axis_off()
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)
    plt.close()
    return ani


def plot_general_skeleton_kmeans(parents, centroid_indices, k, joints, title, dataset, figsize=(7, 7), fps=120, radius=5, title_wrap=20):
    matplotlib.use('Agg')
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, k)]
    n_frames=len(centroid_indices)
    title = '\n'.join(wrap(title, title_wrap))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=None)
        ax2.bar(range(n_frames), height=1, color=[colors[centroid_indices[i]] for i in range(len(joints))], width=1, align='edge', edgecolor="none")                   
        ax2.set_xticks(range(0, n_frames, n_frames//5))
        ax2.yaxis.set_visible(False)
        ax2.axis('off')

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    data = joints.copy().reshape(len(joints), -1, 3)
    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    # elif dataset in ['truebones']: 
    #     data *= 0.2
    elif dataset in ['humanml', 'truebones', 'humanml_mat']:
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=20, ncols=1)
    plt.tight_layout()
    ax = fig.add_subplot(gs[3:-1], projection='3d')  # animation
    ax2 = fig.add_subplot(gs[1])  # color bar
    #ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    n_frames = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]


    def update(index):
        index = min(n_frames-1, int(index*fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        for joint, parent in enumerate(parents[1:], start=1):
            color = colors[centroid_indices[index]]
            ax.plot3D(data[index, [joint, parent], 0], data[index, [joint, parent], 1], data[index, [joint, parent], 2], color=color, solid_capstyle='round', linewidth=3.0)

        
        plt.axis('off')
        ax.set_axis_off()
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        	
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # indicate time marker on colorbar
        if index > 0:
            ax2.axvline(x=index-1, color=colors[centroid_indices[index]],  ymax=1)  # delete previous time marker
        ax2.axvline(x=index, color='black', ymax=1)  # draw time marker

        
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)

    plt.close()
    return ani 

def plot_general_skeleton_pca(parents, pca_results, joints, title, dataset, figsize=(7, 7), fps=120, radius=5):
    matplotlib.use('Agg')
    cmap = plt.get_cmap('tab20')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=None)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    data = joints.copy().reshape(len(joints), -1, 3)
    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    # elif dataset in ['truebones']: 
    #     data *= 0.2
    elif dataset in ['humanml', 'truebones', 'humanml_mat']:
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    n_frames = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]


    def update(index):
        index = min(n_frames-1, int(index*fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        for joint, parent in enumerate(parents[1:], start=1):
            color = pca_results[index]
            ax.plot3D(data[index, [joint, parent], 0], data[index, [joint, parent], 1], data[index, [joint, parent], 2], color=color, solid_capstyle='round')

        
        plt.axis('off')
        ax.set_axis_off()
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        	
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)

    plt.close()
    return ani 

def save_multiple_samples(out_path, file_name,  animations, fps, max_frames):
    sample_save_path = os.path.join(out_path, file_name)
    print(f'saving {file_name}')

    clips = clips_array(animations)
    clips.duration = max_frames/fps
    
    # import time
    # start = time.time()
    clips.write_videofile(sample_save_path, fps=fps, threads=4, logger=None)
    # print(f'duration = {time.time()-start}')
    
    for clip in clips.clips: 
        # close internal clips. Does nothing but better use in case one day it will do something
        clip.close()
    clips.close()  # important
    
def save_sample(out_path, file_name, animation, fps, max_frames):
    sample_save_path = os.path.join(out_path, file_name)
    print(f'saving {file_name}')
    animation.duration = max_frames/fps
    animation.write_videofile(sample_save_path, fps=fps, threads=4, logger=None)
    animation.close()

def plot_general_skeleton_3d_motion(save_path, parents, joints, title, dataset="truebones", figsize=(7, 7), fps=120, radius=5, face_joints = [], fc = None, title_wrap=20):
    ani = get_general_skeleton_3d_motion(parents, joints, title, dataset, figsize, fps, radius, face_joints, fc, title_wrap=title_wrap)
    path = Path(save_path)
    out_dir = path.parent
    fname = path.name
    save_sample(out_dir, fname, ani, fps, len(joints))