import torch
import torch.nn.functional as F

from typing import Union, Optional, Tuple
from jaxtyping import Float

from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.models.layers.attention import Attention


def rotate_queries_or_keys(
    x: Float[torch.Tensor, "batch heads seq dim"],
    pos: Float[torch.Tensor, "seq"],
) -> Float[torch.Tensor, "batch heads seq dim"]:
    B, num_heads, N, D = x.size()

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin().repeat(1, 1, 1, 2)
    emb_cos = freq.cos().repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)

    return (x * emb_cos) + (y * emb_sin)


class RopeAttention(Attention):

    def __init__(self, cfg: Union[dict, HookedViTConfig], layer_id: Optional[int] = None):
        super().__init__(cfg, layer_id)

        self.grid_size = cfg.image_size // cfg.patch_size
        self.grid_depth = cfg.video_num_frames // cfg.video_tubelet_depth

        self.d_dim = int(2 * ((cfg.d_head // 3) // 2))
        self.h_dim = int(2 * ((cfg.d_head // 3) // 2))
        self.w_dim = int(2 * ((cfg.d_head // 3) // 2))

    def get_position_ids(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        total_tokens = int(self.grid_depth * self.grid_size * self.grid_size)
        ids = torch.arange(total_tokens, device=device)

        tokens_per_frame = self.grid_size * self.grid_size
        tokens_per_row = self.grid_size

        frame_ids = ids // tokens_per_frame
        spatial_ids = ids - tokens_per_frame * frame_ids
        height_ids = spatial_ids // tokens_per_row
        width_ids = spatial_ids - tokens_per_row * height_ids

        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(
        self,
        qk: Float[torch.Tensor, "batch heads seq d_head"],
        pos_ids: Tuple[torch.Tensor, ...],
    ) -> Float[torch.Tensor, "batch heads seq d_head"]:
        d_mask, h_mask, w_mask = pos_ids
        s = 0

        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        if s < self.cfg.d_head:
            qkr = qk[..., s:]
            return torch.cat([qkd, qkh, qkw, qkr], dim=-1)
        return torch.cat([qkd, qkh, qkw], dim=-1)

    def calculate_attn_scores(
        self,
        q: Float[torch.Tensor, "batch pos head_index d_head"],
        k: Float[torch.Tensor, "batch pos head_index d_head"],
        attention_mask: Optional[Float[torch.Tensor, "batch pos pos"]] = None,
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3)

        pos_ids = self.get_position_ids(q.device)
        q_t = self.apply_rotary_embeddings(q_t, pos_ids)
        k_t = self.apply_rotary_embeddings(k_t, pos_ids)

        q = q_t.permute(0, 2, 1, 3)
        k = k_t.permute(0, 2, 1, 3)

        return super().calculate_attn_scores(q, k, attention_mask)
