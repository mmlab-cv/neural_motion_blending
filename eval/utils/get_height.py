import numpy as np

# Canonical 23-joint simplified skeleton (Mixamo standard, root = Pelvis/Hips)
_CORPS_NAMES = [
    'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
    'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
]

# End-effectors in _CORPS_NAMES; indices 0 and 2 are the two feet used for height.
_EE_NAMES = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']

# Pre-computed ee indices in the simplified skeleton (stable, no BVH needed)
_EE_ID = [_CORPS_NAMES.index(name) for name in _EE_NAMES]


def _strip_namespace(name):
    """Remove Mixamo namespace prefix (e.g. 'mixamo:Hips' -> 'Hips')."""
    return name[name.find(':') + 1:] if ':' in name else name


def _build_simplified_skeleton(anim, joint_names):
    """
    Map the full BVH skeleton onto the canonical simplified skeleton.

    Returns
    -------
    corps : list[int]
        Index into the full skeleton for each canonical joint, in _CORPS_NAMES order.
    topo : tuple[int]
        Parent index within the simplified skeleton for each canonical joint.
    offsets : np.ndarray [n_corps, 3]
        Bone offsets for the simplified joints (same units as anim.offsets).
    """
    clean_names = [_strip_namespace(n) for n in joint_names]
    name_to_full = {n: i for i, n in enumerate(clean_names)}

    corps = []
    for cname in _CORPS_NAMES:
        idx = name_to_full.get(cname)
        if idx is not None:
            corps.append(idx)

    # simplify_map: full-skeleton index -> simplified index (or -1)
    simplify_map = [-1] * len(joint_names)
    for simp_idx, full_idx in enumerate(corps):
        simplify_map[full_idx] = simp_idx

    # Topology over the simplified skeleton
    raw_parents = anim.parents
    topo = []
    for full_idx in corps:
        p = raw_parents[full_idx]
        topo.append(simplify_map[p] if (p >= 0 and simplify_map[p] >= 0) else 0)
    topo[0] = 0  # root is its own parent

    offsets = anim.offsets[corps]
    return corps, tuple(topo), offsets


def get_height(anim, joint_names):
    """
    Estimate character height as the sum of bone-chain lengths from root to
    the left foot (ee_id[0]) and from root to the right foot (ee_id[2]) in
    the simplified skeleton.

    Parameters
    ----------
    anim        : Animation object with .offsets [n_joints, 3] and .parents
    joint_names : list[str] — joint names aligned with anim joints

    Returns
    -------
    height : float  (same units as anim.offsets)
    """
    _, topo, offsets = _build_simplified_skeleton(anim, joint_names)

    def _chain_length(start):
        res = 0.0
        p = start
        while p != 0:
            res += float(np.dot(offsets[p], offsets[p]) ** 0.5)
            p = topo[p]
        return res

    # Left-foot chain + right-foot chain
    return _chain_length(_EE_ID[0]) + _chain_length(_EE_ID[2])
