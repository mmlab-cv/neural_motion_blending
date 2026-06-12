"""
bvh_io.py — Safe BVH loading utilities.

BVH.load pre-allocates arrays based on the 'Frames: N' header and crashes with
an IndexError when motion2motion writes extra (often corrupt) frames beyond that
count, or with a ValueError when those extra frames have the wrong number of
values.  _bvh_load_safe corrects the mismatch before handing off to BVH.load.
"""

import os
import re
import tempfile

import BVH


def bvh_load_safe(bvh_path):
    """Drop-in replacement for BVH.load that tolerates frame-count mismatches.

    motion2motion occasionally appends extra frames beyond the declared
    'Frames: N' count — typically one corrupt trailing frame with the wrong
    number of values.  We count actual data lines and either truncate (if
    actual > declared) or patch the header down (if actual < declared) before
    calling BVH.load on a temporary corrected copy.

    Returns the same (anim, joint_names, frametime) tuple as BVH.load.
    """
    with open(bvh_path, 'r') as f:
        text = f.read()

    motion_start = text.find('MOTION')
    if motion_start == -1:
        return BVH.load(bvh_path)

    header_part = text[:motion_start]
    motion_lines = text[motion_start:].splitlines()

    frames_line_idx = None
    data_start_idx  = None
    for idx, line in enumerate(motion_lines):
        if re.match(r'\s*Frames\s*:', line):
            frames_line_idx = idx
        elif re.match(r'\s*Frame\s*Time\s*:', line):
            data_start_idx = idx + 1
            break

    if frames_line_idx is None or data_start_idx is None:
        return BVH.load(bvh_path)

    declared  = int(re.search(r'\d+', motion_lines[frames_line_idx]).group())
    data_lines = [l for l in motion_lines[data_start_idx:] if l.strip()]
    actual    = len(data_lines)

    if actual == declared:
        return BVH.load(bvh_path)

    if actual > declared:
        # Truncate extra frames (often corrupt trailing output from motion2motion).
        kept         = data_lines[:declared]
        motion_lines = motion_lines[:data_start_idx] + kept
    else:
        # Fewer frames than declared — patch the header count down.
        motion_lines[frames_line_idx] = re.sub(
            r'\d+', str(actual), motion_lines[frames_line_idx], count=1
        )

    patched = header_part + '\n'.join(motion_lines)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bvh', delete=False) as tmp:
        tmp.write(patched)
        tmp_path = tmp.name
    try:
        result = BVH.load(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result
