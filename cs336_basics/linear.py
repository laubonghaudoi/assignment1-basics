import torch
from jaxtyping import Array, Float
import einx
from torch.nn.init import trunc_normal_


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W: Float[Array, "out_features in_features"] = torch.nn.Parameter(
          torch.randn((out_features, in_features), device=device, dtype=dtype)
        )
        trunc_normal_(self.W, std=0.02)

    def forward(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return einx.dot(
            "d_out d_in, ... d_in -> ... d_out",
            self.W, x
        )


if __name__ == "__main__":
    linear = Linear(10, 20)
    X: Float[Array, "10 10"] = torch.randn(10, 10)
    print(linear(X))
