import torch
from pytorch3d.transforms import rotation_6d_to_matrix, so3_relative_angle # pip install pytorch3d

def avg_per_frame_dist(motion1, motion2, norm):
    # [n_frames, n_features]
    min_len = min(motion1.shape[0], motion2.shape[0])
    if norm == 'fro':
        n_frames = motion1.shape[0]
        return torch.norm(motion2[:min_len] - motion1[:min_len], p='fro').cpu().numpy() / n_frames
    elif norm == 'l2':
        return torch.norm(motion2[:min_len] - motion1[:min_len], p=2, dim=-1).mean().cpu().numpy()
    elif norm == 'loc':
        n_joints = motion1.shape[-1] // 3
        return torch.norm(motion2[:min_len].view(-1, n_joints, 3) - motion1[:min_len].view(-1, n_joints, 3), p=2, dim=-1).mean().cpu().numpy()
    elif norm == 'rot':
        n_joints = motion1.shape[-1] // 6
        motion1_matrices = rotation_6d_to_matrix(motion1[:min_len].view(min_len, n_joints, 6)).view(-1, 3, 3)
        motion2_matrices = rotation_6d_to_matrix(motion2[:min_len].view(min_len, n_joints, 6)).view(-1, 3, 3)
        # Compute relative angles for each joint
        angles = so3_relative_angle(
            motion1_matrices,  # [n_frames1, n_frames2, n_joints, 3, 3]
            motion2_matrices,  # [n_frames1, n_frames2, n_joints, 3, 3]
            cos_angle=False
        ).view(min_len, n_joints)  # [n_frames1, n_frames2, n_joints]
        return angles.mean()
    else:
        raise ValueError(f'invalid nort type [{norm}]')

def pos_avg_l2(motion1, motion2):
    n_joints = motion1.shape[-1] // 3
    return torch.norm(motion2.view(*motion2.shape[:2], n_joints, 3) - motion1.view(*motion1.shape[:2], n_joints, 3), p=2, dim=-1).mean(dim=-1).cpu().numpy()

def pos_avg_cosine_distance(motion1, motion2):
    """
    Compute the angular distance matrix between two motions represented in 6D format.

    Args:
        motion1: Tensor of shape [n_frames1, 1, n_joints * 6].
        motion2: Tensor of shape [1, n_frames2, n_joints * 6].

    Returns:
        dist: Tensor of shape [n_frames1, n_frames2] containing the average angular distance.
    """
    # Reshape motions to [n_frames, n_joints, 6]
    n_joints = motion1.shape[-1] // 6
    motion1 = motion1.view(motion1.shape[0], 1, n_joints, 6).cuda()  # [n_frames1, 1, n_joints, 6]
    motion2 = motion2.view(1, motion2.shape[1], n_joints, 6).cuda()  # [1, n_frames2, n_joints, 6]

    # Convert 6D representations to rotation matrices
    motion1_matrices = rotation_6d_to_matrix(motion1)  # [n_frames1, 1, n_joints, 3, 3]
    motion2_matrices = rotation_6d_to_matrix(motion2)  # [1, n_frames2, n_joints, 3, 3]

    # Tile motion matrices to have matching dimensions for so3_relative_angle
    motion1_matrices = motion1_matrices.repeat(1, motion2_matrices.shape[1], 1, 1, 1)  # [n_frames1, n_frames2, n_joints, 3, 3]
    motion2_matrices = motion2_matrices.repeat(motion1_matrices.shape[0], 1, 1, 1, 1)  # [n_frames1, n_frames2, n_joints, 3, 3]
    out_shape = motion1_matrices.shape[:3]
    motion1_matrices = motion1_matrices.view(-1, 3, 3)
    motion2_matrices = motion2_matrices.view(-1, 3, 3)

    # Compute relative angles for each joint
    angles = so3_relative_angle(
        motion1_matrices,  # [n_frames1, n_frames2, n_joints, 3, 3]
        motion2_matrices,  # [n_frames1, n_frames2, n_joints, 3, 3]
        cos_angle=False
    ).view(out_shape)  # [n_frames1, n_frames2, n_joints]

    # Average over joints to get the distance matrix
    dist = angles.mean(dim=-1)  # [n_frames1, n_frames2]

    return dist.cpu()