import copy

import torch 
import torch.nn as nn
from typing import Dict, List, Optional, Union, Callable, Tuple
from torch import Tensor
import torch.nn.functional as F
CUDA_LAUNCH_BLOCKING=1

class GraphRelEmbedder(nn.Module):
    """Helper to manage the various graph-relation embeddings."""
    def __init__(self, d_model, max_path_len, value_emb=False, v_joints_rel=False):
        super().__init__()
        self.value_emb = value_emb

        # topological relations:
        # 0 (self/none)
        # 1...max_path_len (skeleton hops)
        # max_path_len+1 (real <-> virtual) 'abstract topo rel'
        # max_path_len+2 (virtual <-> virtual) 'virtual neighbor topo rel'
        num_topo_types = max_path_len + 3 if v_joints_rel else max_path_len + 1
        self.topo_q = nn.Embedding(num_topo_types, d_model)
        self.topo_k = nn.Embedding(num_topo_types, d_model)
        
        # edge relationships:
        # 0-5 (skeleton relations: none, parent, child, sibling, etc.)
        # 6 (real <-> virtual) 'abstract edge'
        # 7 (virtual <-> virtual) 'virtual neighbor edge'
        num_edge_types = 6 if not v_joints_rel else 8
        self.edge_q = nn.Embedding(num_edge_types, d_model)
        self.edge_k = nn.Embedding(num_edge_types, d_model)
        
        if value_emb:
            self.topo_v = nn.Embedding(num_topo_types, d_model)
            self.edge_v = nn.Embedding(num_edge_types, d_model)

    def forward(self):
        # Returns weights for the spatial attention pass
        v_weights = (self.topo_v.weight, self.edge_v.weight) if self.value_emb else (None, None)
        return self.topo_q.weight, self.edge_q.weight, self.topo_k.weight, self.edge_k.weight, *v_weights

class GraphMultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout, nheads, max_path_len=5, value_emb=False, v_joints_rel=False):
        super().__init__()

        # Graph Attention
        self.nheads = nheads
        self.att_size = att_size = d_model // nheads
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(d_model, nheads * att_size)
        self.linear_k = nn.Linear(d_model, nheads * att_size)
        self.linear_v = nn.Linear(d_model, nheads * att_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(nheads * att_size, d_model)

        # Spatial attention
        self.graph_embed = GraphRelEmbedder(d_model, max_path_len, value_emb, v_joints_rel)

    def forward(
        self,
        q, k, v,
        distance,
        edge_attr,
        mask=None,
    ):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.nheads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.nheads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.nheads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)  # [b, h, k_len, d_k]

        query_hop_emb, query_edge_emb, \
        key_hop_emb, key_edge_emb, \
        value_hop_emb, value_edge_emb = self.graph_embed()

        sequence_length = v.shape[2]
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]

        # NOTE: input chunking
        query_hop_emb = query_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        query_edge_emb = query_edge_emb.view(
            1, -1, self.nheads, self.att_size
        ).transpose(1, 2)
        key_hop_emb = key_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        key_edge_emb = key_edge_emb.view(
            1, num_edge_types, self.nheads, self.att_size
        ).transpose(1, 2)

        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = torch.gather(
            query_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )
        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = torch.gather(
            query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )

        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = torch.gather(
            key_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )
        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = torch.gather(
            key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )

        spatial_bias = (query_hop + key_hop)
        edge_bais = (query_edge + key_edge)

        x = torch.matmul(q, k.transpose(2, 3)) + spatial_bias + edge_bais

        x = x * self.scale

        if mask is not None:
            x = x + mask

        x = torch.softmax(x, dim=3)
        x = self.dropout(x)
        if value_hop_emb is not None:
            value_hop_emb = value_hop_emb.view(
                1, num_hop_types, self.nheads, self.att_size
            ).transpose(1, 2)
            value_edge_emb = value_edge_emb.view(
                1, num_edge_types, self.nheads, self.att_size
            ).transpose(1, 2)

            value_hop_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_hop_types),
                device=value_hop_emb.device,
            )
            value_hop_att = torch.scatter_add(
                value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )
            value_edge_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_edge_types),
                device=value_hop_emb.device,
            )
            value_edge_att = torch.scatter_add(
                value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )
        x = torch.matmul(x, v)
        if value_hop_emb is not None:
            x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.nheads * d_v)

        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class SpatialAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, max_path_len=5, value_emb=False, v_joints_rel=False):
        super().__init__()
        self.model = GraphMultiHeadAttention(d_model, dropout, nhead, max_path_len, value_emb, v_joints_rel=v_joints_rel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : Tensor, topo_rel: Tensor, edge_rel: Tensor, mask : Optional[Tensor]) -> Tensor:
        """ all joints in some frame exchange information based on their values, accounting for topological and edge relationships. """
        F, B, J, C = x.shape
        x = x.view(F * B, J, C)
        topo_rel = topo_rel.unsqueeze(0).repeat(F, 1, 1, 1).view(-1, J, J)
        edge_rel = edge_rel.unsqueeze(0).repeat(F, 1, 1, 1).view(-1, J, J)
        out = self.model(x, x, x, topo_rel, edge_rel, mask)
        return self.dropout(out.reshape(F, B, J, C))

class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.model = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        """ Each joint attends to itself across time, within a specified time window. """
        F, B, J, C = x.shape
        x = x.view(F, B * J, C)
        attn_out, _ = self.model(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout(attn_out.view(F, B, J, C))

class ConditionAttention(nn.Module):
    """
    Unified Conditioning Layer.
    - "semantic_modulation": Identity-Gating to avoid the 1.0 attention collapse. Requires latent_codes == 1.
    - "mha": Frame-Local Multihead Attention to search V latents.
    - "film": FiLM-style affine modulation (scale + shift) from a single latent. Requires latent_codes == 1.
    """
    STRATEGIES = ("semantic_modulation", "mha", "film")

    def __init__(self, latent_codes: int, d_model: int, num_heads: int, dropout: float = 0.1, strategy: str = "semantic_modulation"):
        super().__init__()
        assert strategy in self.STRATEGIES, f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}."
        if strategy in ("semantic_modulation", "film"):
            assert latent_codes == 1, f"Strategy '{strategy}' requires latent_codes == 1, got {latent_codes}."
        self.strategy = strategy
        self.latent_codes = latent_codes
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        if strategy == "semantic_modulation":
            self.query_projector = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
            self.gate_gen = nn.Sigmoid()
            self.value_map = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
        elif strategy == "mha":
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        elif strategy == "film":
            self.film_proj = nn.Linear(d_model, d_model * 2)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        """
        x: (T, B, J, C) -> Joints
        memory: (T, B, V, C) -> Conditioning latents (V == 1 for semantic_modulation / film)
        """
        if self.strategy == "semantic_modulation":
            return self._forward_gated(x, memory)
        elif self.strategy == "mha":
            return self._forward_basic(x, memory)
        else:
            return self._forward_film(x, memory)

    def _forward_gated(self, x: Tensor, memory: Tensor) -> Tensor:
        """
        x: (T, B, J, C) - Target Joints
        memory: (T, B, 1, C) - Source Latent
        """
        T, B, J, C = x.shape
        m = memory.expand(-1, -1, J, -1)
        combined = torch.cat([x, m], dim=-1)
        gate = self.gate_gen(self.query_projector(combined))
        v = self.value_map(m)
        return self.dropout(self.out_proj(v * gate))

    def _forward_basic(self, x: Tensor, memory: Tensor) -> Tensor:
        """
        V > 1 Strategy: Search across V latents within the frame.
        x:      (T, B, J, C)
        memory: (T, B, V, C)
        """
        T, B, J, C = x.shape
        V = memory.shape[2]
        q = x.permute(2, 0, 1, 3).reshape(J, T * B, C)
        kv = memory.permute(2, 0, 1, 3).reshape(V, T * B, C)

        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv
        )

        attn_out = attn_out.view(J, T, B, C).permute(1, 2, 0, 3)
        return self.dropout(attn_out)

    def _forward_film(self, x: Tensor, memory: Tensor) -> Tensor:
        """
        FiLM: affine modulation of x by a single conditioning latent.
        x:      (T, B, J, C)
        memory: (T, B, 1, C)
        """
        T, B, J, C = x.shape
        gamma, beta = self.film_proj(memory).chunk(2, dim=-1)  # each (T, B, 1, C)
        return self.dropout(gamma.expand(-1, -1, J, -1) * x + beta.expand(-1, -1, J, -1))

class GraphMotionDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None): 
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = decoder_layer.d_model

    def forward(self,
            x: Tensor,
            topo_rel: Tensor,
            edge_rel: Tensor,
            timesteps_embs: Optional[Tensor] = None,
            memory: Optional[Tensor] = None,
            spatial_mask: Optional[Tensor] = None,
            temporal_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            get_layer_activation : List[int] = [],
        ) -> Union[Tensor , Dict[str, Tensor]]:

        activations = dict()
        output = x
        for layer_ind, mod in enumerate(self.layers):            
            output = mod(
                output, topo_rel, edge_rel,
                timesteps_emb=timesteps_embs,
                memory=memory,
                spatial_mask=spatial_mask, 
                temporal_mask=temporal_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask, 
                memory_key_padding_mask=memory_key_padding_mask, 
            )
            if layer_ind in get_layer_activation:
                activations[layer_ind] = output.clone()
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output, activations


class GraphMotionDecoderLayer(nn.Module):
    def __init__(self,
            d_model: int = 128,
            num_heads: int = 4,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            activation: Callable = F.relu,
            max_path_len: int = 5,
            value_emb: bool = False,
            timestep: bool = True,
            attend_latents: int = 1,
            build_latents: bool = False,
            second_temporal: bool = False,
            condition_strategy: str = "semantic_modulation",
        ):
        super().__init__()
        assert not (attend_latents > 0 and build_latents), "Current implementation does not support building and attending to latents within the same layer."
        self.d_model = d_model
        needs_cross_attn = attend_latents > 0
        if isinstance(activation, str):
            act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}
            activation = act_map.get(activation.lower(), nn.ReLU)

        # Blocks & Norms
        self.spatial_attn = SpatialAttention(d_model, num_heads, dropout, max_path_len, value_emb, build_latents)
        self.spatial_norm = nn.LayerNorm(d_model)

        self.temporal_attn = TemporalAttention(d_model, num_heads, dropout)
        self.temporal_norm = nn.LayerNorm(d_model)

        self.cross_attn = ConditionAttention(attend_latents, d_model, num_heads, dropout, strategy=condition_strategy) if needs_cross_attn else None
        self.cross_norm = nn.LayerNorm(d_model) if needs_cross_attn else None
        
        self.temporal_attn_2 = TemporalAttention(d_model, num_heads, dropout) if second_temporal else None
        self.temporal_norm_2 = nn.LayerNorm(d_model) if second_temporal else None

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        # Support Modules
        self.timestep_embed = nn.Linear(d_model, d_model) if timestep else None

    def forward(self,
            x: Tensor,
            topo_rel: Tensor,
            edge_rel: Tensor,
            timesteps_emb: Optional[Tensor] = None,
            memory: Optional[Tensor] = None,
            spatial_mask: Optional[Tensor] = None,
            temporal_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None, # for future use
        ) -> Tensor:
        # x.shape (frames, bs, njoints, feature_len)
        bs = x.shape[1]
        if self.timestep_embed is not None:
            x = x + self.timestep_embed(timesteps_emb).view(1, bs, 1, self.d_model)
        x = self.spatial_norm(x + self.spatial_attn(x, topo_rel, edge_rel, spatial_mask))
        x = self.temporal_norm(x + self.temporal_attn(x, temporal_mask, tgt_key_padding_mask))
        if self.cross_attn is not None:
            x = self.cross_norm(x + self.cross_attn(x, memory))
        if self.temporal_attn_2 is not None:
            x = self.temporal_norm_2(x + self.temporal_attn_2(x, temporal_mask, tgt_key_padding_mask))
        x = self.ffn_norm(x + self.ffn(x))
        return x