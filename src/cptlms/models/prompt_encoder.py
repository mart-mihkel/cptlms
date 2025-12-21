import logging
from typing import Annotated, Literal

import torch
from torch import Tensor
from torch.nn import LSTM, Embedding, Linear, Module, ReLU, Sequential

logger = logging.getLogger(__name__)

type EncoderReparameterizationType = Literal["emb", "mlp", "lstm"]


class PromptEncoder(Module):
    """
    Prompt encoder from the [p-tuning paper](https://arxiv.org/abs/2103.10385)
    """

    reparam_type: EncoderReparameterizationType

    def __init__(
        self,
        token_dim: int,
        num_virtual_tokens: int,
        hidden_size: int,
        reparam_type: EncoderReparameterizationType = "mlp",
    ) -> None:
        assert reparam_type in ["emb", "mlp", "lstm"], (
            "Invalid encoder type, must be emb, mlp or lstm!"
        )

        super().__init__()

        self.prompt_tokens = torch.arange(num_virtual_tokens)
        self.embedding = Embedding(num_virtual_tokens, token_dim)

        self.reparam_type = reparam_type  # type: ignore
        if reparam_type == "mlp":
            self.mlp_head = Sequential(
                Linear(token_dim, hidden_size),
                ReLU(),
                Linear(hidden_size, hidden_size),
                ReLU(),
                Linear(hidden_size, token_dim),
            )
        elif reparam_type == "lstm":
            self.lstm_head = LSTM(
                input_size=token_dim,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.0,
                bidirectional=True,
                batch_first=True,
            )

            self.mlp_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size * 2, token_dim),
            )

    def forward(self) -> Annotated[Tensor, "1 virtual token"]:
        device = self.embedding.weight.device
        prompt_tokens = self.prompt_tokens.to(device)

        embeds = self.embedding(prompt_tokens).unsqueeze(0)
        if self.reparam_type == "lstm":
            embeds = self.mlp_head(self.lstm_head(embeds)[0])
        elif self.reparam_type == "mlp":
            embeds = self.mlp_head(embeds)

        return embeds
