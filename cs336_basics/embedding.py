import torch
from jaxtyping import Float
from torch.nn.init import trunc_normal_


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.W: Float[torch.Tensor, "num_embeddings embedding_dim"] = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

    def forward(self, token_ids: torch.Tensor) -> Float[torch.Tensor, "..."]:
        return self.W[token_ids]


if __name__ == "__main__":
    embedding = Embedding(10, 20)
    print(embedding(torch.tensor([0, 1, 2])))
