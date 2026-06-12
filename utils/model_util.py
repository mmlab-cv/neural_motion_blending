import torch
import torch.nn as nn

from model.anytop import AnyTop
from model.motion_diffusion_ae import MoDiffAE
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


def load_model(model, state_dict):
    """ Loads model weights """
    expected_missing_keys = ['clip_model.']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0, f"Unexpected keys found when loading the model: {unexpected_keys}"
    assert all([k.startswith(s) for k in missing_keys for s in expected_missing_keys])


def create_model_and_diffusion_general_skeleton(args):
    """ Creates the model and diffusion process based on the provided arguments. """
    if args.model_name == 'AnyTop':
        print("Initializing [AnyTop]...")
        model = AnyTop(**get_gmdm_args(args))
    elif args.model_name == 'MoDiffAE':
        print("Initializing [MoDiffAE]...")
        model = MoDiffAE(**get_modiffae_args(args))

    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def get_gmdm_args(args):
    """ AnyTop args formatting. """
    t5_model_dim = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }
    # default args
    t5_out_dim = t5_model_dim[args.t5_name]
    njoints = 23
    nfeats = 1
    max_joints=143
    feature_len=13
    cond_mode = 'object_type'

    return {'njoints': njoints, 'nfeats': nfeats, 't5_out_dim': t5_out_dim,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'max_joints': max_joints, 
            'feature_len':feature_len,  'skip_t5': args.skip_t5, 'value_emb': args.value_emb, 'root_input_feats': 13}

def get_modiffae_args(args):
    
    t5_model_dim = {
    "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }
    # default args
    t5_out_dim = t5_model_dim[args.t5_name]
    njoints = 23
    nfeats = 1
    max_joints=143
    feature_len=13

    return {
            'njoints': njoints, 'nfeats': nfeats, 'max_joints': max_joints, 'root_input_feats': 13, 'feature_len':feature_len,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_heads': 4, 'dropout': 0.1, 'activation': "gelu",
            't5_out_dim': t5_out_dim, 'skip_t5': args.skip_t5,
            'value_emb': args.value_emb, 'max_path_len': 5,
            'layers_stochast': args.num_layers_stochastic,
            'layers_semantic': args.num_layers_semantic,
            'num_virtual_joints': args.num_virtual_joints,
            'compress_virtual_joints': args.compress_virtual_joints,
            'projection_head_depth': args.projection_head_depth,
            'kl_bottleneck': args.kl_bottleneck,
            'second_temporal_attn': args.second_temporal_attn,
            'semantic_feat_mode': args.semantic_feat_mode,
            'condition_strategy': args.condition_strategy,
    }
    
    
def create_gaussian_diffusion(args):
    """ Creates Gaussian diffusion """
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 100
    scale_beta = 1.  # no scaling
    
    sampler_type = getattr(args, 'sampler', '')
    if sampler_type == 'ddim':
        ddim_steps = args.ddim_steps if args.ddim_steps > 0 else steps
        timestep_respacing = f'ddim{ddim_steps}'
    else:
        timestep_respacing = ''

    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_fs=args.lambda_fs,
        lambda_geo=args.lambda_geo,
        lambda_kl=args.lambda_kl,
    )

def summarize_model(module, name="root", depth=0, is_last=True, prefix=""):
    if depth == 0:
        total_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print("\n" + "-" * 50)
        print(f"Model Summary | Trainable Parameters: {total_trainable_params / 1e6:.2f}M")
        print("-" * 50)

    connector = "└── " if is_last else "├── "
    
    dims = "---"
    if isinstance(module, (nn.Linear, nn.LazyLinear)):
        dims = f"[{module.in_features}, {module.out_features}]"
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        dims = f"[{module.in_channels}, {module.out_channels}, {module.kernel_size}]"
    elif isinstance(module, nn.Embedding):
        dims = f"[{module.num_embeddings}, {module.embedding_dim}]"
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        dims = f"[{module.normalized_shape if hasattr(module, 'normalized_shape') else module.num_features}]"
    elif isinstance(module, nn.MultiheadAttention):
        dims = f"[Embed: {module.embed_dim}, Heads: {module.num_heads}]"

    params = list(module.parameters(recurse=False))
    grad_status = ""
    if params:
        requires_grad = any(p.requires_grad for p in params)
        grad_status = f" [Grad: {'✅' if requires_grad else '❌'}]"

    print(f"{prefix}{connector}{name} ({module.__class__.__name__}) | {grad_status} | {dims}")

    new_prefix = prefix + ("    " if is_last else "│   ")
    
    # Show raw Parameters (like virtual joint tokens)
    raw_params = list(module.named_parameters(recurse=False))
    children = list(module.named_children())
    
    for i, (p_name, p) in enumerate(raw_params):
        is_last_param = (i == len(raw_params) - 1) and (len(children) == 0)
        p_conn = "└── " if is_last_param else "├── "
        p_grad = "✅" if p.requires_grad else "❌"
        print(f"{new_prefix}{p_conn}{p_name} (Parameter) | [Grad: {p_grad}] | {list(p.shape)}")

    # Show Sub-Modules
    for i, (child_name, child) in enumerate(children):
        summarize_model(
            child, 
            name=child_name, 
            depth=depth + 1, 
            is_last=(i == len(children) - 1), 
            prefix=new_prefix
        )