# TrueBones evaluation:

## Environment Setup:
Install dependencies using conda and pip:
```
conda install -c conda-forge eigen
pip install git+https://github.com/PeizhuoLi/ganimator-eval-kernel.git
pip install pytorch3d
```

## Running the Evaluation
```
python -m --seed <seed> eval.eval_truebones --eval_gt_dir <gt_motion_dir> --eval_gen_dir <generated_motion_dir> --benchmark_path <benchmark_path> --unique_str <output_fname_suffix>

```
To reproduce the evaluation results reported in Tables 3 and 8, follow these steps:
Generate 20 samples, each 120 frames long, for every benchmark skeleton you plan to evaluate (see eval/benchmarks for the model-specific skeleton lists). Place all generated .npy files in a single directory named _gen_motion_dir_. Do not change the default generation seed.
Run the evaluation command with seed=10.
Note: For technical reasons, the models we’ve released are not identical to those evaluated in the paper, but they achieve comparable results.

### Arguments Description
* `--seed` To evaluate using different seeds. Default value is 10. 
* `--eval_gt_dir` Path to the ground truth motions directory (NumPy .npy files). Default value is dataset/truebones/zoo/truebones_processed/motions/.
* `--eval_gen_dir` Path to the directory containing generated motions (NumPy .npy files).
Note: To reproduce paper results, each character in the benchmark should have 20 motions, each 120 frames long.
* `--benchmark_path` Path to a text file listing character names to evaluate.
For the full AnyToP benchmark, use: eval/benchmarks/benchmark_all.txt. For specific subsets, use: eval/benchmarks/benchmark_<subset>.txt, 
where <subset> is one of: bipeds, quadropeds, millipeds_snakes, or flying.
* `--unique_str` (Optional) A custom suffix added to the evaluation results filename.
The results will be saved as eval_npy_mode_npy_loc_<unique_str>.log in the parent directory of eval_gen_dir. It's recommended to set a unique suffix (e.g., based on seed or experiment name) to avoid overwriting results from previous runs.



