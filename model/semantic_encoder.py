import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat
from .modules.motion_transformer import GraphMotionDecoder, GraphMotionDecoderLayer
from model.anytop import InputProcess

from typing import Optional, Dict, List

class SemanticEncoder(nn.Module):
    def __init__(self,
            num_layers: int = 4,
            latent_dim: int = 128,
            ff_size: int = 1024,
            num_heads: int = 4,
            dropout: float = 0.1,
            activation: str = "gelu",
            max_path_len: int = 5,
            value_emb: bool = False,
            input_process: Optional[InputProcess] = nn.Identity(),
            num_virtual_joints: int = 1,
            projection_head_depth: int = 0,
            compress_virtual_joints: bool = True,
            kl_bottleneck: bool = False,
            semantic_feat_mode: str = 'full',
        ):
        """ Given clean motion x_0 projects frame-wise information into a space-agnostic latent space z_sem """
        super().__init__()
        self.num_virtual_joints = num_virtual_joints

        if semantic_feat_mode not in ('full', 'rots', 'vel'):
            raise ValueError(f"Unknown semantic_feat_mode '{semantic_feat_mode}'. Choose from 'full', 'rots', 'vel'.")
        self.semantic_feat_mode = semantic_feat_mode
        self.max_path_len = max_path_len
        self.num_heads = num_heads

        # modules
        if num_virtual_joints > 0:
            build_latents = True # wether to add virtual joint topo/edge embeds
            self.v_joint_emb = nn.Parameter(torch.randn(num_virtual_joints, latent_dim))
            self._v_joint_init()
        else:
            build_latents = False
            compress_virtual_joints = True
            self.spatial_pooling = MaskedSpatialPooling('avg')

        self.input_process = input_process
        self.backbone = GraphMotionDecoder(
            GraphMotionDecoderLayer(
                d_model=latent_dim, num_heads=num_heads,
                dim_feedforward=ff_size, dropout=dropout, activation=activation,
                max_path_len=max_path_len, value_emb=value_emb, second_temporal=False,
                timestep=False, build_latents=build_latents, attend_latents=-1,
            ),
            num_layers,
            norm=None,
        )

        # semantic projection
        self.compress_virtual_joints = compress_virtual_joints
        out_d = latent_dim if not kl_bottleneck else latent_dim * 2
        in_d = (max(num_virtual_joints, 1) * latent_dim) if compress_virtual_joints else latent_dim
        if projection_head_depth > 0:
            depth = projection_head_depth
            dims = [in_d] + [(in_d + latent_dim) // 2] * (depth - 1) + [out_d]
            
            layers = []
            for i in range(depth):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < depth - 1:
                    layers.extend([nn.GELU(), nn.Dropout(dropout)])
            self.projection_head = nn.Sequential(*layers)
        else:
            self.projection_head = nn.Linear(in_d, out_d)

        self.kl_layer = None if not kl_bottleneck else KLBottleneck(latent_dim)
        
    def forward(self,
            x: Tensor,
            y: Dict[str, Tensor],
            topo_rel: Tensor,
            edge_rel: Tensor,
            j_mask: Tensor,
            t_mask: Tensor,
            tpos_first_frame: Tensor,
            alpha: Optional[float] = None,
            get_layer_activation: Optional[List[int]] = [],
        ):
        # x.shape (bs, ncontrols, njoints, nfeats, nframes)
        C = x.shape[1]

        # input process
        x = rearrange(x, 'b c j f t -> (b c) j f t')
        if self.semantic_feat_mode != 'full':
            feat_mask = torch.zeros(x.shape[2], dtype=x.dtype, device=x.device)
            if self.semantic_feat_mode == 'rots':
                feat_mask[3:9] = 1.0
            elif self.semantic_feat_mode == 'vel':
                feat_mask[9:12] = 1.0
            x = x * feat_mask[None, None, :, None]
        
        x = self.input_process(x, tpos_first_frame, y['joints_names_embs'], y['crop_start_ind'])
        x, j_mask, t_mask, topo_rel, edge_rel = self._add_virtual_joints(x, j_mask, t_mask, topo_rel, edge_rel)

        # backbone
        T, _, J, _ = x.shape
        spatial_mask = 1.0 - j_mask[:, 0, 0, :, :]
        spatial_mask = spatial_mask.unsqueeze(1).unsqueeze(1).repeat(1, T, self.num_heads, 1, 1).reshape(-1, self.num_heads, J, J)
        temporal_mask = 1.0 - t_mask.repeat(1, J, self.num_heads, 1, 1).reshape(-1, T, T).float()
        spatial_mask[spatial_mask == 1.0] = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9
        
        x, activations = self.backbone(
            x, topo_rel, edge_rel,
            timesteps_embs=None, memory=None,
            spatial_mask=spatial_mask, temporal_mask=temporal_mask,
            get_layer_activation=get_layer_activation
        )
        
        # project
        if self.num_virtual_joints == 0:
            # No virtual joints = pool -> project
            z_sem = self.spatial_pooling(x, j_mask)
            z_sem = self.projection_head(z_sem)
            z_sem = rearrange(z_sem, 'f (b c) d -> f b c d', c=C).unsqueeze(2)
        elif self.compress_virtual_joints:
            # Compress virtual joints into a single vector (per frame) during projection
            virtual_joints = x[:, :, -self.num_virtual_joints:, :]
            z_sem = rearrange(virtual_joints, 'f bc v d -> f bc (v d)')
            z_sem = self.projection_head(z_sem)
            z_sem = rearrange(z_sem, 'f (b c) d -> f b c d', c=C).unsqueeze(2) # (J-dim)
        else:
            # Keep virtual joints separate and project each one, resulting in multiple vectors per frame
            virtual_joints = x[:, :, -self.num_virtual_joints:, :]
            z_sem = self.projection_head(virtual_joints)
            z_sem = rearrange(z_sem, 'f (b c) v d -> v f b c d', c=C)
        
        mu, logvar = None, None
        if self.kl_layer:
            z_sem, mu, logvar = self.kl_layer(z_sem)            

        return z_sem, {'mu': mu, 'logvar': logvar}

    def _add_virtual_joints(self, x, j_mask, t_mask, topo_rel, edge_rel):
        """ Updates input with virtual joints ([CLS]-like tokens for pooling global information) """
        if self.num_virtual_joints <= 0:
            return x, j_mask, t_mask, topo_rel, edge_rel

        # Setup Dimensions
        num_frames, batch_size, num_real_j, latent_dim = x.shape
        total_j = num_real_j + self.num_virtual_joints

        # Expand Node Features (x)
        v_joint_emb = repeat(self.v_joint_emb, 'v d -> f b v d', f=num_frames, b=batch_size)
        x = torch.cat([x, v_joint_emb], dim=2)

        # Update Spatial Mask (j_mask)
        j_existence = torch.diagonal(j_mask, dim1=-2, dim2=-1) # [B, 1, 1, J]
        j_existence = torch.nn.functional.pad(j_existence, (0, self.num_virtual_joints), value=1.0)
        
        # Broadcast existence to a full connectivity matrix [B, 1, 1, J+V, J+V]
        j_mask = j_existence.unsqueeze(-1) * j_existence.unsqueeze(-2)
        is_padded = ~(j_existence.squeeze([1,2]).bool()) # [B, J+V]
        pad_rect_mask = is_padded.unsqueeze(1) | is_padded.unsqueeze(2)

        # Update Relationship Matrices (Topology & Edge)
        REL_IDX = {
            'topo_rv': self.max_path_len + 1, # real -> virtual
            'topo_vv': self.max_path_len + 2, # virtual -> virtual
            'edge_rv': 6,                     # real -> virtual
            'edge_vv': 7                      # virtual -> virtual
        }

        def expand_rel_matrix(real_2_real, real_2_virtual, virtual_2_virtual, device, dtype):
            # Initialize with Virtual-to-Virtual default
            new_rel = torch.full((batch_size, total_j, total_j), virtual_2_virtual, 
                                device=device, dtype=dtype)
            
            # Fill sub-regions
            new_rel[:, :num_real_j, :num_real_j] = real_2_real
            new_rel[:, :num_real_j, num_real_j:] = real_2_virtual
            new_rel[:, num_real_j:, :num_real_j] = real_2_virtual
            
            # Cleanup
            diag_indices = torch.arange(total_j, device=device)
            new_rel[:, diag_indices, diag_indices] = 0
            
            # Zero out relations involving padding joints
            new_rel[pad_rect_mask] = 0
            return new_rel

        topo_rel = expand_rel_matrix(topo_rel, REL_IDX['topo_rv'], REL_IDX['topo_vv'], topo_rel.device, topo_rel.dtype)
        edge_rel = expand_rel_matrix(edge_rel, REL_IDX['edge_rv'], REL_IDX['edge_vv'], edge_rel.device, edge_rel.dtype)

        return x, j_mask, t_mask, topo_rel, edge_rel

    def _v_joint_init(self, scale=0.02):
        # they start as orthogonal vectors
        # to promote diversity at training start
        with torch.no_grad():
            nn.init.orthogonal_(self.v_joint_emb)
            self.v_joint_emb.mul_(scale)


class KLBottleneck(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x: Tensor, sample: bool = True):
        mu, logvar = torch.chunk(x, 2, dim=-1)
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        if sample and self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sem = mu + eps * std
        else:
            z_sem = mu

        # remove virtual joint and ncontrols dimensions (for convenience, it's not really necessary)
        mu = mu.squeeze((2,3)) 
        logvar = logvar.squeeze((2,3))

        return z_sem, mu, logvar

class MaskedSpatialPooling(nn.Module):
    def __init__(self, method: str = 'avg'):
        super().__init__()
        self.method = method

    def forward(self, x: Tensor, j_mask: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [num_frames, batch_size, num_joints, latent_dim]
            j_mask: Spatial mask [batch_size, 1, 1, num_joints, num_joints] 
                   where 1.0 is valid, 0.0 is padding.
        Returns:
            Pooled tensor [num_frames, batch_size, latent_dim]
        """
        mask = torch.diagonal(j_mask.squeeze(1).squeeze(1), dim1=-2, dim2=-1)
        mask = mask.unsqueeze(0).unsqueeze(-1)
        
        if self.method == 'avg':
            x_masked = x * mask
            sum_x = x_masked.sum(dim=2)
            count = mask.sum(dim=2).clamp(min=1)
            return sum_x / count

        elif self.method == 'max':
            fill_value = torch.finfo(x.dtype).min
            x_masked = x.masked_fill(mask == 0, fill_value)
            return x_masked.max(dim=2).values

        else:
            raise ValueError(f"Unknown pooling method: {self.method}")