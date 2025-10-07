import torch
from jaxtyping import Array, Float
from torch import Tensor
import torch.nn.functional as F
import einx
from cs336_basics.linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1: Float[Tensor, "d_ff d_model"] = torch.nn.Parameter(
            torch.randn((d_ff, d_model), device=device, dtype=dtype)
        )
        self.w2: Float[Tensor, "d_model d_ff"] = torch.nn.Parameter(
            torch.randn((d_model, d_ff), device=device, dtype=dtype)
        )
        self.w3: Float[Tensor, "d_ff d_model"] = torch.nn.Parameter(
            torch.randn((d_ff, d_model), device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "..."]:
        f1: Float[Tensor, "... d_ff"] = einx.dot("d_ff d_model, ... d_model -> ... d_ff", self.w1, x)
        silu: Float[Tensor, "... d_ff"] = f1 * F.sigmoid(f1)
        w3x: Float[Tensor, "... d_ff"] = einx.dot("d_ff d_model, ... d_model -> ... d_ff", self.w3, x)
        return einx.dot("d_model d_ff, ... d_ff -> ... d_model", self.w2, silu * w3x)
