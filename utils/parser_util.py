from argparse import ArgumentParser
import argparse
import os
import json
import copy
import sys

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_model_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    if isinstance(args.model_path, list) and len(args.model_path) == 1:
        args.model_path = args.model_path[0]
    
    # load args from model
    assert not isinstance(args, list) and not isinstance(args.model_path, list), 'Deprecated feature..'
    args = extract_args(copy.deepcopy(args), args_to_overwrite, args.model_path)

    return args

def extract_args(args, args_to_overwrite, model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

    # backward compatibility
    if isinstance(args.emb_trans_dec, bool):
        if args.emb_trans_dec:
            args.emb_trans_dec = 'cls_tcond_cross_tcond'
        else: 
            args.emb_trans_dec = 'cls_none_cross_tcond'
    return args

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')

def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=10, type=int, help="Batch size during training.")
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=100, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")

def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=4, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=128, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--lambda_fs", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_geo", default=0.0, type=float, help="Geodesic Loss.")
    group.add_argument("--t5_name", default='t5-base', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"], type=str,
                       help="Choose t5 pretrained model")
    group.add_argument("--temporal_window", default=31, type=int,
                       help="temporal window size")
    group.add_argument("--skip_t5", action='store_true',
                       help="If passed, joints names wont be added to features")
    group.add_argument("--value_emb", action='store_true',
                       help="If passed, graph multihead attention learns GRPE value embeddings")
    # NEW
    group.add_argument("--model_name", default='MoDiffAE', choices=['AnyTop', 'MoDiffAE'], type=str, help="Model name. This will only affect the default pretrained_base_model_path, and the architecture of the controlnet if control is enabled.")
    group.add_argument("--cond_mask_prob", default=0.0, type=float, help="The probability of masking the condition during training. For classifier-free guidance learning.")
    group.add_argument("--guidance_scale", default=1.0, type=float, help="scale for classifier-free guidance during training. Only relevant if cond_mask_prob > 0 and control is enabled.")
    ## MoDiffAE
    group.add_argument("--num_layers_semantic", default=4, type=int, help="Number of layers in the semantic encoder of MoDiffAE.")
    group.add_argument("--num_layers_stochastic", default=4, type=int, help="Number of layers in the stochastic decoder of MoDiffAE.")
    group.add_argument("--num_virtual_joints", default=5, type=int, help="Number of virtual joints to be added to the input of the semantic encoder. Virtual joints are connected to all other joints and can be used by the model to route information across the skeleton.")
    group.add_argument("--compress_virtual_joints", default=True, type=bool, help="If True, the output of the semantic encoder will be compressed by projecting the features of the virtual joints, before being fed to the stochastic decoder. If False, the features of the virtual joints will be kept separate and fed to the stochastic decoder as is.")
    group.add_argument("--projection_head_depth", default=2, type=int, help="Depth of the MLP projection head applied on top of the semantic encoder output. If 0, no projection head is applied and the output of the semantic encoder is used as is.")

    group.add_argument("--second_temporal_attn", action='store_true', help="If True, a second temporal attention layer will be added to each layer of the motion transformer blocks in both the semantic encoder and the stochastic decoder of MoDiffAE. The second temporal attention will be applied after the cross-attention (if cross-attention is enabled) and will attend only to the current motion features, without attending to the latents. This can be used to enhance temporal modeling in the model, at the expense of increased computational cost.")
    group.add_argument("--kl_bottleneck", action='store_true', help="If True, a KL divergence bottleneck will be applied on the semantic latent space of MoDiffAE. If enabled, the projection head of the semantic encoder will output both mean and log-variance, which will be used to sample the latent code fed to the stochastic decoder. During training, a KL divergence loss will be applied between the distribution of the sampled latent code and a standard normal distribution, weighted by lambda_kl.")
    group.add_argument("--lambda_kl", default=0.0, type=float, help="Weight for KL divergence loss on the semantic latent space of MoDiffAE.")
    group.add_argument("--kl_warmup_steps", default=25_005, type=int, help="Number of warmup steps for KL divergence loss. During warmup, the weight of the KL divergence is zero.")
    group.add_argument("--kl_anneal_steps", default=25_005, type=int, help="Number of steps for KL divergence loss annealing. During annealing, the weight of the KL divergence loss will be linearly increased from zero to lambda_kl.")
    group.add_argument("--semantic_feat_mode", default='full', choices=['full', 'rots', 'vel'], type=str,
                       help="Feature subset fed to the semantic encoder. 'full' uses all 13 features; "
                            "'rots' keeps only the 6D rotation features (indices 3-8); "
                            "'vel' keeps only the velocity features (indices 9-11). "
                            "Unused features are zeroed out.")
    group.add_argument("--condition_strategy", default='semantic_modulation', choices=['semantic_modulation', 'mha', 'film'], type=str,
                       help="Conditioning strategy used in ConditionAttention within the stochastic decoder. "
                            "'semantic_modulation' uses identity-gating (requires num_latent_codes == 1); "
                            "'mha' uses frame-local multi-head cross-attention; "
                            "'film' uses FiLM affine modulation (requires num_latent_codes == 1).")

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default="truebones", choices=["truebones", "mixamo"], type=str,
                       help="Dataset to use for training.")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--objects_subset", default='all', choices=['all', 'quadropeds' , 'flying', 'bipeds', 'millipeds', 'millipeds_snakes', 'quadropeds_clean', 'millipeds_clean', 'flying_clean', 'bipeds_clean', 'all_clean'], type=str,
                       help="Object subset (truebones only).")

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--model_prefix", type=str,
                       help="Unique string at the beggining of the model name.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--ml_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")

    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=50, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=25_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=500_019, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=120, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--use_ema", action='store_true',
                       help="If True, will use EMA model averaging.")
    group.add_argument("--balanced", action='store_true',
                       help="Use balancing sampler for fairness between topologies")
    # ADDED
    group.add_argument("--model_summary", action='store_true',
                       help="If True, will print model summary before training and exit. No training will be performed.")
    group.add_argument("--gen_alpha", default=[0.0, 0.5, 1.0], type=float, nargs='+', 
                        help="Alpha values for generating motions during training. Mixing will be performed for each alpha specified in the list.")

def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--cond_path", default='', type=str,
                       help="provide cond.py path in case you wish to generate motion for skeleton not included in Truebones dataset.")
    group.add_argument("--render_video", default=True, type=bool,
                       help="If True, will render an mp4 video for each generated motion. Requires ffmpeg to be installed.")
    group.add_argument("--dataset_npy_dir", default='', type=str, help="Path to the directory containing the npy motion files. If omitted, inferred from --dataset (e.g. dataset/mixamo_processed/motions).")

    # diff. sampling options
    group.add_argument("--sampler", default='ddpm', choices=['ddpm', 'ddim'], type=str, help="Sampling strategy to use for mixing.")
    group.add_argument("--ddim_steps", default=-1, type=int, help="Number of steps to use for DDIM sampling. Only relevant if sampler is set to ddim. Negative to use all steps.")
    group.add_argument("--ddim_inversion", action='store_true', help="If True, will perform DDIM inversion to obtain noise latents for the reference motion, before performing mixing. Only relevant if sampler is set to ddim.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--object_type", default=['Flamingo'], type=str, nargs='+',
                       help="An object type to be generated. If empty, will generate flamingo :).")
    
def add_mix_options(parser):
    group = parser.add_argument_group('mix')

    group.add_argument("--ref_motion", type=str, default='dataset/mixamo_processed/motions/Kaya_Male Sitting Pose_19882.npy') # Comodoa___Attack1_211 Raptor2___EggTend_699
    group.add_argument("--tgt_motion", type=str, default='dataset/mixamo_processed/motions/Aj_Male Sitting Pose_4070.npy') #Gazelle___Fall_377

    group.add_argument("--ref_samples", type=str, default='', help="Path to txt file containing reference motion paths (one per line). If specified, will override ref_motion argument.")
    group.add_argument("--tgt_samples", type=str, default='', help="Path to txt file containing target motion paths (one per line). If specified, will override tgt_motion argument.")
    group.add_argument("--blend_preset", type=str, default='', help="Path to .json file containing predefined blending parameters. If specified, will override all other blending related arguments and set them to predefined values according to the chosen preset. e.g. linear ease-in-out with 20 frames overlap, etc, (for benchmarking especially).")

    group.add_argument("--ref_frame_start", default=0, type=int)
    group.add_argument("--tgt_frame_start", default=0, type=int)
    
    group.add_argument("--ref_num_loops", default=1, type=int, help="how many times to repeat REF animation")
    group.add_argument("--tgt_num_loops", default=1, type=int, help="how many times to repeat TGT animation")

    group.add_argument("--control_mode", default="both", choices=["ref", "tgt", "both"], type=str, help="Whether to use the reference motion, the target motion, or both as control signal for the generation.")
    group.add_argument("--blend_schedule", default='ease', choices=['static', 'linear', 'ease'], type=str, help="When using both motion as controls, determines how to module the frame-wise intersection between the 2 signals in terms of alpha values.")
    group.add_argument("--interpolation_mode", default='lerp', choices=['lerp', 'slerp'], type=str, help="How control signals are mixed.")
    group.add_argument("--alpha_values", default=[1.0], type=float, nargs='+', help="When using 2 controls in static blend mode, the value of Alpha for the interpolation")
    group.add_argument("--overlap_length", default=-1, type=int, help="Number of frames for the intersection between the 2 controls. if negative it will be ignored and frame_starts are used")
    group.add_argument("--intersection_only", action='store_true', help="When set, generate only the overlapping (intersection) frames between ref and tgt controls, instead of the full union span.")
    group.add_argument("--ease_slope", default=1.0, type=float, help="slope for the sine ease in/out blend mode. set < 1.0 for flatter slope, > 1.0 for sharper")
    group.add_argument("--transition_slerp", action='store_true', help="If set, SLERP between inverted ref/tgt noise in the transition zone instead of using pure Gaussian noise.")
    group.add_argument("--resume", action='store_true', help="If set, skip ref-tgt pairs that already have complete outputs (checked via JSON dumps). Only valid when --ref_samples/--tgt_samples are provided.")
    
    # ./dataset/truebones/zoo/truebones_processed/motions
    # Monkey___Run_579
    # Monkey___Walk_580
    # Puppy_Puppy_Run_668
    # Puppy_Puppy_Walk_669
    # Chicken_Chicken_IdlePecking_209
    # Raptor2___IdleForward_708
    # Flamingo_Flamingo_OneLEgBEnt_353
    # Flamingo_Flamingo_BendIdle_352
    # Flamingo_Flamingo_Walk_355

    # ./assets/
    # Monkey___B2Attack_574.npy
    # Scorpion___SlowForward_837.npy


def add_matching_options(parser):
    group = parser.add_argument_group('matching')
    # Samples we want to match
    group.add_argument("--sample_ref", default='./dataset/truebones/zoo/truebones_processed/motions/Ostrich___Walk_591.npy', type=str)
    group.add_argument("--sample_tgt", default=['./dataset/truebones/zoo/truebones_processed/motions/Flamingo_Flamingo_Walk_355.npy'], type=str, nargs='+')
    # Which feats to use for matching and how to perform the matching
    group.add_argument("--feats", default='t5_proj', choices=['t5', 't5_proj', 'tpose', 'sum', 'enrichment', 'dift'], type=str)
    group.add_argument("--strategy", default='softmax', choices=['softmax', 'sinkhorn'], type=str)
    # parameters for the matching strategies
    group.add_argument("--softmax_tau", default=0.01, type=float, help="Temperature for softmax matching.")
    group.add_argument("--sinkhorn_epsilon", default=0.05, type=float, help="Epsilon for sinkhorn matching.")
    group.add_argument("--sinkhorn_n_iter", default=100, type=int, help="Number of iterations for sinkhorn matching.")
    group.add_argument("--dift_layer", default=[0], type=int, nargs='+', help="Layer to extract DIFT features from.")
    group.add_argument("--dift_timestep", default=[90], type=int, nargs='+', help="Timestep to extract DIFT features from.")

def add_dift_options(parser):
    # bvhs_dir, sample_bvh, face_joints, save_dir=None, tpos_bvh=None
    group = parser.add_argument_group('dift')
    # group.add_argument("--apply_pca", action='store_true',
    #                    help="apply pca on feats before calculating similarity.")
    group.add_argument("--sample_ref", default='./assets/Monkey___B2Attack_574.npy', type=str,
                       help="sample bvh ref.")
    group.add_argument("--sample_tgt", default=['./assets/Scorpion___SlowForward_837.npy'], type=str, nargs='+',
                       help="sample bvh tgt.")
    group.add_argument("--tmp_save_dir", default='', type=str,
                       help="temporal save dir.")
    group.add_argument("--suffix", default='', type=str,
                       help="file suffix.")
    group.add_argument("--dift_type", default='spatial', choices=['spatial', 'temporal'], type=str,
                       help="apply dift on spatial or temporal features")
    group.add_argument("--layer", default=0, type=int,
                       help="Layer to extract DIFT features from.")
    group.add_argument("--timestep", default=90, type=int,
                       help="Timestep to extract DIFT features from.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")
    group.add_argument("--samples", default=['assets/Ostrich___Attack_581.npy'], type=str, nargs='+',
                    help="samples npy")
    group.add_argument("--object_type", default='Flamingo', type=str,
                    help="An object type to be generated. If empty, will generate flamingo :).")
    group.add_argument("--upper_body_root", default=[0], type=int, nargs='+',
                       help="defines the root joints of the upper body for upper_body editing mode.")
    group.add_argument("--unique_str", default='', type=str, help="A string to be added to the file name to identify a specific change. Should start with '_'.")


def add_render_options(parser):
    group = parser.add_argument_group('render')
    group.add_argument('--bvh_path', type=str, default='assets/Truebones_Chicken', help='path of animation bvh file')
    group.add_argument('--save_dir', type=str, default='save/render_out', help='')
    group.add_argument('--scale', type=float, default=0.7, help='')
    group.add_argument('--cylinder_radius', type=float, default=0.42, help='')
    group.add_argument('--sphere_radius', type=float, default=0.56, help='')
    group.add_argument("--subset", default='bipeds', choices=['quadropeds' , 'flying', 'bipeds', 'millipeds_snakes'], type=str, help="Object subset.")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--source", default='npy', type=str, choices=['npy', 'bvh'], help="File format to load for both generated samples and GT: 'npy' reads processed .npy files, 'bvh' reads .bvh files.")
    group.add_argument("--feats", default='rot', type=str, choices=['rot', 'loc'], help="Feature type: 'rot' uses 6D joint rotations; 'loc' uses joint positions (FK-derived for bvh, stored for npy).")
    group.add_argument("--root_relative", action='store_true', help="If set, subtract root joint position from all joints (loc only). Removes global trajectory; root joint is excluded from output.")
    group.add_argument("--benchmark_path", default='eval/benchmarks/benchmark_all.txt', type=str,  help="Path to benchmark character names. If empty, will use all excluding the characters_to_exclude")
    group.add_argument("--eval_gt_dir", default='dataset/truebones/zoo/truebones_processed/motions', type=str, help="Path to gt dir.")
    group.add_argument("--eval_gen_dir", required=True, type=str, help="Path to gen dir.")
    group.add_argument("--characters_to_exclude", default='MouseyNoFingers,Mousey_m,Trex,SabreToothTiger,Raptor2', type=str, help="Comma separated list of characters to exclude. The default is character with more than 40 motions.")
    group.add_argument("--unique_str", default='', type=str, help="A string to be added to the file name to identify a specific change. Should start with '_'.")
    group.add_argument("--ref_samples", default=None, type=str, help="Path to ref samples txt file (benchmark ref list). Used to prioritise relevant GT clips in the pool.")
    group.add_argument("--tgt_samples", default=None, type=str, help="Path to tgt samples txt file (benchmark tgt list). Used to prioritise relevant GT clips in the pool.")
    group.add_argument("--exclude_self_pairs", action='store_true', help="Skip blend triples where ref and tgt are the same clip. Useful for static/intersection benchmarks.")

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_training_options(parser)
    return parser.parse_args()

def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    return args

def mix_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_mix_options(parser)
    args = parse_and_load_from_model(parser)
    return args

def process_new_skeleton_args():
    parser = ArgumentParser()
    group = parser.add_argument_group('process_new_skeleton')
    group.add_argument("--object_name", required=True, type=str,
                       help="A character's indicative name")
    group.add_argument("--bvh_dir", required=True, type=str,
                       help="Path to a directory containing BVH files of the skeleton. More files improve statistical accuracy for \
                           motion denormalization.")
    group.add_argument("--save_dir", required=True, type=str,
                       help="Output directory.")
    group.add_argument("--face_joints_names", default=["RLeg1", "LLeg1", "RArm1", "LArm1"], type=str, nargs=4,
                       help="Four joints defining skeleton orientation ([right hip, left hip, right shoulder, left shoulder] or equivalent). \
                           Used to align the skeleton to Z+ and XZ plane.")
    group.add_argument("--tpos_bvh", default='', type=str,
                       help="A BVH file of the character's natural rest pose for meaningful rotation learning. \
                            If missing, the code selects a pose from the provided BVH files.")
    args = parser.parse_args()
    return args

def dift_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_dift_options(parser)
    args = parse_and_load_from_model(parser)

    return args

def match_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_matching_options(parser)
    args = parse_and_load_from_model(parser)
    return args


def edit_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    add_base_options(parser)
    add_evaluation_options(parser)
    return parser.parse_args()

def render_parser():
    parser = ArgumentParser()
    add_render_options(parser)
    if "--" not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(argv)

