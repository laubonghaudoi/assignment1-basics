from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE as RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
import torch
import einx
from jaxtyping import Float, Int


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, **kwargs):
        super().__init__()
        # d_k = d_v = d_model // num_heads
        self.w_qkv = Linear(d_model, 3 * d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

        self.num_heads: int = num_heads
        self.d_model: int = d_model
        self.d_head: int = d_model // num_heads

    def forward(
        self,
        x: Float[torch.Tensor, "... sequence_length d_in"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_length: int = x.shape[-2]
        # "3*d_k d_in, ... sequence_length d_in -> ... sequence_length 3*d_k"
        qkv: Float[torch.Tensor, "... sequence_length 3*d_model"] = self.w_qkv(x)

        # Split into separate q, k, v tensors

        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape from (batch, sequence_length, dim) to (batch, heads, sequence_length, head_dim)
        q = einx.rearrange(
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head",
            q,
            num_heads=self.num_heads,
        )
        k = einx.rearrange(
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head",
            k,
            num_heads=self.num_heads,
        )
        v = einx.rearrange(
            "... sequence_length (num_heads d_head) -> ... num_heads sequence_length d_head",
            v,
            num_heads=self.num_heads,
        )

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        # Create causal mask for self-attention
        mask = ~torch.triu(
            torch.ones((sequence_length, sequence_length), device=x.device, dtype=torch.bool), diagonal=1
        )

        y: Float[torch.Tensor, "... num_heads sequence_length d_head"] = scaled_dot_product_attention(q, k, v, mask)
        y = einx.rearrange("... num_heads sequence_length d_head -> ... sequence_length (num_heads d_head)", y)
        return self.output_proj(y)
