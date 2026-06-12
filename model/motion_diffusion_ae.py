from dataclasses import dataclass

import torch

import torch.nn as nn
from typing import Optional, Dict, List

from .anytop import InputProcess, OutputProcess, create_sin_embedding
from .semantic_encoder import SemanticEncoder
from .stochastic_decoder import StochasticDecoder

torch.cuda.empty_cache()

@dataclass  
class ControlConfig:
    x: torch.Tensor                   # [batch_size, n_controls, njoints, nfeats, max_frames]
    y: Dict                           # dict of conditioning information for the control signal
    alpha: torch.Tensor = 0.0         # [0.0, 1.0] interpolation factor between control signals (if n_controls=2)
    blend_mask: torch.Tensor = 0.0    # mask for temporal blending
    interpolation_mode: str = "lerp"  # interpolation strategy to combine control signals (e.g. 'lerp'. 'slerp', etc.)

def gather_vars(x, y):
    """
    NOTE: the collate adds extra joint/frame dim which may/may not be used
    e.g. in the original AnyTop the extra frame is used (for the tpose, always concatenated as new first frame), but the extra joint is dropped.
    y : {
        'joints_mask': (bs, 1, 1, max_joints+1, max_joints+1)
        'mask': (bs, 1, 1, max_frames+1, max_frames+1)
        'graph_dist': (bs, max_joints, max_joints)
        'joints_relations': (bs, max_joints, max_joints)
        'tpos_first_frame': (bs, max_joints, features_len)
    }
    """
    j_mask = y['joints_mask'].to(x.device)
    t_mask = y['mask'].to(x.device)
    topo_rel = y['graph_dist'].long().to(x.device)
    edge_rel = y['joints_relations'].long().to(x.device)
    tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)
    
    # drop the extra joint
    # NOTE: in AnyTop it's done hidden in the forward, we do it here to avoid confusion
    j_mask = j_mask[:, :, :, 1:, 1:]
    
    # "fix" t-pose temporal mask
    # Check -> https://github.com/Anytop2025/Anytop/issues/34
    t_mask = t_mask.clone() 
    t_mask[:, :, :, 0, 0] = True
    t_mask[:, :, :, 0, 1:] = False

    return topo_rel, edge_rel, j_mask, t_mask, tpos_first_frame


class MoDiffAE(nn.Module):
    def __init__(self,
        max_joints,
        feature_len,
        latent_dim=128,
        ff_size=1024,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        layers_stochast=4,
        layers_semantic=4,
        compress_virtual_joints=True,
        **kargs
    ):
        super().__init__()

        self.max_joints = max_joints
        self.feature_len = feature_len
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.layers_stochast = layers_stochast
        self.layers_semantic = layers_semantic
        self.compress_virtual_joints = compress_virtual_joints
        self.root_input_feats = kargs.get('root_input_feats', 13)
        self.t5_out_dim = kargs.get('t5_out_dim', 512)
        self.skip_t5=kargs.get('skip_t5', False)
        self.value_emb=kargs.get('value_emb', False)
        self.max_path_len = kargs.get('max_path_len', 5)
        self.num_virtual_joints = kargs.get('num_virtual_joints', 3)
        self.projection_head_depth = kargs.get('projection_head_depth', 1)
        self.cond_mode = kargs.get('cond_mode', 'clean_motion')
        self.input_feats = self.feature_len
        self.kl_bottleneck = kargs.get('kl_bottleneck', False)
        self.second_temporal_attn = kargs.get('second_temporal_attn', False)
        self.semantic_feat_mode = kargs.get('semantic_feat_mode', 'full')
        self.condition_strategy = kargs.get('condition_strategy', 'semantic_modulation')
        self.num_latent_codes = self.num_virtual_joints if not self.compress_virtual_joints else 1

        self.semantic_encoder = SemanticEncoder(
            self.layers_semantic, self.latent_dim, self.ff_size, self.num_heads, self.dropout, self.activation, self.max_path_len, self.value_emb,
            InputProcess(self.input_feats, self.root_input_feats, self.latent_dim, self.t5_out_dim, skip_t5=self.skip_t5),
            self.num_virtual_joints, projection_head_depth=self.projection_head_depth, compress_virtual_joints=self.compress_virtual_joints, kl_bottleneck=self.kl_bottleneck,
            semantic_feat_mode=self.semantic_feat_mode,
        )
        self.stochastic_decoder = StochasticDecoder(
            self.layers_stochast, self.latent_dim, self.ff_size, self.num_heads, self.dropout, self.activation, self.max_path_len, self.value_emb, self.second_temporal_attn,
            InputProcess(self.input_feats, self.root_input_feats, self.latent_dim, self.t5_out_dim, skip_t5=self.skip_t5),
            self.num_latent_codes, condition_strategy=self.condition_strategy,
        )
        self.output_process = OutputProcess(self.feature_len, self.root_input_feats, self.max_joints, self.latent_dim)

    def forward(self,
        x: torch.Tensor,
        timesteps : torch.Tensor,
        y: Optional[Dict] = None,
        control: Optional[ControlConfig] = None,
        get_layer_activation: Optional[List[int]] = [],
    ):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        
        # Semantic encoder
        if 'z_sem' not in y.keys():
            z_sem, semantic_dict = self.semantic_encoder(
                control.x, control.y,
                *gather_vars(control.x, control.y),
                alpha=control.alpha,
                get_layer_activation=[]
            )
            z_sem = self._mix(z_sem, control) # latent mixing (if any)
            #z_sem = torch.nn.functional.normalize(z_sem, dim=-1) # normalized latent
        else:
            # caching option
            z_sem = y['z_sem']
            semantic_dict = {}

        # Stochastic decoder
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        output, activations = self.stochastic_decoder(
            x, y, timesteps_emb, z_sem,
            *gather_vars(x, y),
            get_layer_activation=get_layer_activation
        )

        # final projection
        output = self.output_process(output)

        # ret.
        out_dict = {'out': output, 'z_sem': z_sem, **semantic_dict}
        db_dict = {'activations': activations}
        return out_dict, db_dict

    def _mix(self, z_sem, control):
        """ If multiple controls are given, mix them in the latent space by linear interpolation. """
        ncontrols = control.x.shape[1]
        
        if ncontrols == 1:
            return z_sem.squeeze(3)

        elif ncontrols == 2: 
            T_local, B, _, _, C = z_sem.shape
            T_global = control.alpha.shape[0]
            device = z_sem.device

            # Expand alpha and mask to include the T-pose frame at index 0
            full_alpha = torch.cat([torch.tensor([0.0], device=device), control.alpha]).view(-1, 1, 1, 1)
            full_mask = torch.cat([torch.tensor([True], device=device), control.blend_mask]).view(-1, 1, 1, 1)

            # Initialize global latents [T_global + 1, B, 1, C]
            z_ref_g = torch.zeros((T_global + 1, B, 1, C), device=device)
            z_tgt_g = torch.zeros((T_global + 1, B, 1, C), device=device)

            # T-pose (always index 0)
            z_ref_g[0] = z_sem[0, :, :, 0]
            z_tgt_g[0] = z_sem[0, :, :, 1]
            
            for i in range(2):
                # Calculate intended start/end in the GLOBAL canvas
                # We shift by +1 for the T-pose at index 0
                out_s = control.y['crop_start_ind'][i] + 1
                out_e = out_s + control.y['lengths'][i]
                
                # CLAMP indices to T_global + 1 to prevent out-of-bounds
                # This handles the case where mode='tgt' but ref is much longer
                safe_out_e = min(out_e, T_global + 1)
                actual_len = safe_out_e - out_s
                
                if actual_len > 0:
                    if i == 0: # Reference
                        z_ref_g[out_s:safe_out_e] = z_sem[1 : 1 + actual_len, :, :, 0]
                    else: # Target
                        z_tgt_g[out_s:safe_out_e] = z_sem[1 : 1 + actual_len, :, :, 1]

            # Mix and Mask
            z_mixed = torch.lerp(z_ref_g, z_tgt_g, full_alpha)
            return z_mixed * full_mask
       
        else:
            raise NotImplementedError("Only 1 or 2 control signals are supported")
    
    def _apply(self, fn):
        super()._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
