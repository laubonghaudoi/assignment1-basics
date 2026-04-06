import math
import torch
from torch import Tensor
import einx
from jaxtyping import Float, Bool
from cs336_basics.multihead_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RoPE


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.d_ff: int = d_ff
        self.max_seq_len: int = max_seq_len
        self.theta: float = theta

        self.self_attn: CausalMultiHeadSelfAttention = CausalMultiHeadSelfAttention(d_model, num_heads)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, "batch sequence_length d_model"],
        rope: RoPE | None,
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        x = x + self.self_attn(self.ln1(x), rope)
        x = x + self.ffn(self.ln2(x))
        return x
