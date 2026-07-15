import torch
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, scaling: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = scaling

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_kv_groups > 1:
            batch, _, seq, head_dim = k.shape
            
            k = k.unsqueeze(2)
            k = k.expand(batch, self.num_kv_heads, self.num_kv_groups, seq, head_dim)
            k = k.reshape(batch, self.num_heads, seq, head_dim)
            v = v.unsqueeze(2)
            v = v.expand(batch, self.num_kv_heads, self.num_kv_groups, seq, head_dim)
            v = v.reshape(batch, self.num_heads, seq, head_dim)
            

        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            scale=self.scaling
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        return attn_output