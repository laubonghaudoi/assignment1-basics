import torch
from jaxtyping import Array, Float
from torch import Tensor
import einx
from torch.nn.init import trunc_normal_


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.weight: Float[Tensor, " d_model"] = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        trunc_normal_(self.weight, std=0.02)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype

        RMS = torch.sqrt(einx.mean("... d_model -> ... 1", x**2) + self.eps)

        result: Float[Tensor, "..."] = x * self.weight / RMS

        return result.to(in_dtype)
