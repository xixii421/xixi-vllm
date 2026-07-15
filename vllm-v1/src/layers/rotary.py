import torch
from torch import nn

def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        
        assert rotary_dim % 2 == 0, "rotary_dim must be even"
        
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)


    def _rotary(self, x: torch.Tensor, positions: torch.Tensor):
        positions_flat = positions.flatten().to(self.cos_sin_cache.device)
        cos_sin = self.cos_sin_cache.index_select(0, positions_flat)
        
        cos_sin = cos_sin.to(dtype=x.dtype)
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        x_shape = x.shape
        n_tokens = positions_flat.numel()
        
        x_reshaped = x.reshape(n_tokens, -1, self.head_size)
        
        x_rot = _apply_rotary_emb(x_reshaped[..., :self.rotary_dim], cos, sin)
        
        if self.rotary_dim >= self.head_size:
            x_out = x_rot
        else:
            x_out = torch.cat((x_rot, x_reshaped[..., self.rotary_dim:]), dim=-1)
            
        return x_out.reshape(x_shape)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        return self._rotary(query, positions), self._rotary(key, positions)