# 🏋️ Training

`train/train_anytop.py` trains on Truebones. `--model_name` selects the architecture:
`AnyTop` (base cross-topology diffusion transformer) or `MoDiffAE` (default — semantic-encoder /
stochastic-decoder model used for blending).

Make sure the dataset is processed first (see [DATA.md](DATA.md)).

## 🧠 MoDiffAE — the paper's model

MoDiffAE has two pooling variants, set via `--num_virtual_joints`: **global-pool**
(`--num_virtual_joints 0`, pools all joint tokens directly) and **attention-pool**
(`--num_virtual_joints 5` default, routes through learned virtual joint tokens instead). **The
model reported in the paper is the global-pool variant** (`truebones_globpool` in `save/`). To
reproduce it:

```bash
python -m train.train_anytop \
    --model_name MoDiffAE --dataset truebones --objects_subset all \
    --batch_size 10 --latent_dim 128 --layers 4 --lr 1e-4 --num_steps 450000 \
    --num_virtual_joints 0 --lambda_geo 1.0 --balanced \
    --save_dir save/modiffae_truebones_all_globpool
```

(`--lambda_geo 1.0` enables the geodesic auxiliary loss, as used for this checkpoint.) The
attention-pool variant (`truebones_attnpool`) is trained the same way, just drop
`--num_virtual_joints 0` to fall back to the default of 5.

## 🌱 AnyTop base model

```bash
python -m train.train_anytop \
    --model_name AnyTop --dataset truebones --objects_subset all \
    --batch_size 16 --latent_dim 128 --layers 4 \
    --model_prefix anytop_all
```

## 📝 Notes

`--objects_subset` restricts characters (`all`, `quadropeds`, `bipeds`, `flying`,
`millipeds_snakes`, ...). Checkpoints/config are written to `<save_dir>/model######.pt` + `args.json`
every `--save_interval` steps (if `--save_dir` is omitted it defaults to
`save/<model_prefix>_dataset_<dataset>_bs_<bs>_latentdim_<dim>`). Resume with
`--resume_checkpoint <path/to/model######.pt> --save_dir <same dir> --overwrite`.
