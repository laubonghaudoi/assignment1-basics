import torch
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
import einx


class Softmax(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.dim: int = dim

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        input_max = x.max(dim=self.dim, keepdim=True)[0]
        return F.softmax(x - input_max, dim=self.dim)
