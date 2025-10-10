import torch
from jaxtyping import Float
import einx
from torch.nn.init import trunc_normal_


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.W: Float[torch.Tensor, "out_features in_features"] = torch.nn.Parameter(
            torch.randn((out_features, in_features), device=device, dtype=dtype)
        )
        trunc_normal_(self.W, std=0.02)

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return einx.dot("d_out d_in, ... d_in -> ... d_out", self.W, x)


if __name__ == "__main__":
    linear = Linear(10, 20)
    X: Float[torch.Tensor, "10 10"] = torch.randn(10, 10)
    print(linear(X))
