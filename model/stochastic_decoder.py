import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from .modules.motion_transformer import GraphMotionDecoder, GraphMotionDecoderLayer
from model.anytop import InputProcess

from typing import Optional, Dict, List

class StochasticDecoder(nn.Module):
    def __init__(self,
            num_layers: int = 4,
            latent_dim: int = 128,
            ff_size: int = 1024,
            num_heads: int = 4,
            dropout: float = 0.1,
            activation: str = "gelu",
            max_path_len: int = 5,
            value_emb: bool = False,
            second_temporal_attn: bool = False,
            input_process: Optional[InputProcess] = nn.Identity(),
            num_latent_codes: int = 1,
            condition_strategy: str = "semantic_modulation",
        ):

        """ Given noisy motion x_t at timestep t and the latent space z_sem from the semantic encoder, predicts x_0. """
        super().__init__()
        self.num_heads = num_heads
        self.num_latent_codes = num_latent_codes

        # modules
        self.input_process = input_process
        self.backbone = GraphMotionDecoder(
            GraphMotionDecoderLayer(
                d_model=latent_dim, num_heads=num_heads,
                dim_feedforward=ff_size, dropout=dropout, activation=activation,
                max_path_len=max_path_len, value_emb=value_emb, second_temporal=second_temporal_attn,
                timestep=True, build_latents=False, attend_latents=num_latent_codes,
                condition_strategy=condition_strategy,
            ),
            num_layers,
            norm=None
        )

    def forward(self,
            x: Tensor,
            y: Dict[str, Tensor],
            timesteps_embs: Tensor,
            z_sem : Tensor,
            topo_rel: Tensor,
            edge_rel: Tensor,
            j_mask: Tensor,
            t_mask: Tensor,
            tpos_first_frame: Tensor,
            get_layer_activation: Optional[List[int]] = [],
            
        ):
        # x.shape (bs, njoints, nfeats, nframes)

        # input process
        x = self.input_process(x, tpos_first_frame, y['joints_names_embs'], y['crop_start_ind'])

        # backbone
        T, _, J, _ = x.shape
        spatial_mask = 1.0 - j_mask[:, 0, 0, :, :]
        spatial_mask = spatial_mask.unsqueeze(1).unsqueeze(1).repeat(1, T, self.num_heads, 1, 1).reshape(-1, self.num_heads, J, J)
        temporal_mask = 1.0 - t_mask.repeat(1, J, self.num_heads, 1, 1).reshape(-1, T, T).float()
        spatial_mask[spatial_mask == 1.0] = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9
        
        x, activations = self.backbone(
            x, topo_rel, edge_rel,
            timesteps_embs=timesteps_embs, memory=z_sem,
            spatial_mask=spatial_mask, temporal_mask=temporal_mask,
            get_layer_activation=get_layer_activation
        )

        return x, activations

        
