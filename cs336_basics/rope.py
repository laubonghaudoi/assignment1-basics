import torch
from jaxtyping import Float, Int


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.d_k: int = d_k
        self.theta: float = theta
        self.max_seq_len: int = max_seq_len

        assert self.d_k % 2 == 0, f"d_k must be even, got {self.d_k}"
        self.cos_cache: Float[torch.Tensor, "seq_len d_k_half"] | None = None
        self.sin_cache: Float[torch.Tensor, "seq_len d_k_half"] | None = None

        self._build_cache()

    def _build_cache(self):
        positions: Float[torch.Tensor, "seq_len"] = torch.arange(self.max_seq_len, dtype=torch.float32)
        dim_indices: Float[torch.Tensor, "d_k_half"] = torch.arange(self.d_k // 2, dtype=torch.float32)
        freq: Float[torch.Tensor, "d_k_half"] = 1.0 / (self.theta ** (2 * dim_indices / self.d_k))
        angles: Float[torch.Tensor, "seq_len d_k_half"] = torch.outer(positions, freq)

        self.sin_cache: Float[torch.Tensor, "seq_len d_k_half"] = torch.sin(angles)
        self.cos_cache: Float[torch.Tensor, "seq_len d_k_half"] = torch.cos(angles)

    def forward(
        self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        Apply rotary position embeddings to the input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token position indices of shape (..., seq_len)

        Returns:
            Tensor with RoPE applied, same shape as input
        """
        # Get the sin and cos values for the given positions
        # token_positions shape: (..., seq_len)
        # We need to index into our caches which have shape (max_seq_len, d_k // 2)
        cos: Float[torch.Tensor, "... seq_len d_k_half"] = self.cos_cache[token_positions]  # (..., seq_len, d_k // 2)
        sin: Float[torch.Tensor, "... seq_len d_k_half"] = self.sin_cache[token_positions]  # (..., seq_len, d_k // 2)

        # Split x into two halves: the even and odd indices
        # x has shape (..., seq_len, d_k)
        # We reshape to (..., seq_len, d_k // 2, 2) to work with pairs
        x_reshaped: Float[torch.Tensor, "... seq_len d_k_half 2"] = x.reshape(
            *x.shape[:-1], -1, 2
        )  # (..., seq_len, d_k // 2, 2)
        x1: Float[torch.Tensor, "... seq_len d_k_half"] = x_reshaped[..., 0]  # (..., seq_len, d_k // 2) - even indices
        x2: Float[torch.Tensor, "... seq_len d_k_half"] = x_reshaped[..., 1]  # (..., seq_len, d_k // 2) - odd indices

        # Apply rotation:
        # [x1']   [cos  -sin] [x1]
        # [x2'] = [sin   cos] [x2]
        x1_rotated: Float[torch.Tensor, "... seq_len d_k_half"] = x1 * cos - x2 * sin
        x2_rotated: Float[torch.Tensor, "... seq_len d_k_half"] = x1 * sin + x2 * cos

        # Stack them back together and reshape to original shape
        x_rotated: Float[torch.Tensor, "... seq_len d_k"] = torch.stack(
            [x1_rotated, x2_rotated], dim=-1
        )  # (..., seq_len, d_k // 2, 2)
        x_rotated: Float[torch.Tensor, "... seq_len d_k"] = x_rotated.reshape(*x.shape)  # (..., seq_len, d_k)

        return x_rotated
