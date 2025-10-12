import math
import torch
from torch import Tensor
import einx
from jaxtyping import Float, Bool


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    d_k: int = Q.shape[-1]

    scores: Float[Tensor, "... queries keys"] = einx.dot(
        "... queries d_k, ... keys d_k -> ... queries keys", Q, K
    ) / math.sqrt(d_k)

    if mask is not None:
        float_mask = torch.where(mask, 0.0, float("-inf"))
        scores: Float[Tensor, "... queries keys"] = scores + float_mask

    attention_weights = torch.softmax(scores, dim=-1)

    return einx.dot("... queries keys, ... keys d_v -> ... queries d_v", attention_weights, V)
