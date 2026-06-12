import torch


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def char_from_npy_stem(stem):
    """
    Extract the character name from a .npy filename stem.

    Handles both regular characters (e.g. 'Claire_Walking_42' -> 'Claire')
    and Mixamo homeomorphic variants whose names contain an underscore
    (e.g. 'Goblin_m_Walking_42' -> 'Goblin_m').

    The convention is: if the token immediately after the first '_' is 'm',
    the character name is '<first>_m'; otherwise it is just '<first>'.
    """
    parts = stem.split('_')
    if len(parts) > 1 and parts[1].split('_')[0] == 'm':
        return parts[0] + '_m'
    return parts[0]


def freeze_joints(x, joints_to_freeze):
    # Freezes selected joint *rotations* as they appear in the first frame
    # x [bs, [root+n_joints], joint_dim(6), seqlen]
    frozen = x.detach().clone()
    frozen[:, joints_to_freeze, :, :] = frozen[:, joints_to_freeze, :, :1]
    return frozen
