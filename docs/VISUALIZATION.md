# 🎥 Visualization

Stick-figure `.mp4` renders are produced automatically alongside every dataset/`sample/mix.py`
output (skip with `--render_video False`; requires `ffmpeg`) — no separate step needed.

For a higher-quality Blender render of one or more BVH files:

```bash
blender --background --python visualization/bvh2skeleton.py -- \
    --bvh_path assets/truebones/Coyote___Attack3_224.bvh --save_dir save/render_out --subset quadropeds
```

`--bvh_path` accepts a single file or a directory (e.g. `assets/truebones/` to render all bundled
sample clips at once). Outputs one `.blend` file per input under `--save_dir`.
