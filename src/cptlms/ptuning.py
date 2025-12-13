import logging
from typing import Annotated, Literal, cast

import torch
from torch import FloatTensor, Tensor
from torch.nn import LSTM, Embedding, Linear, Module, ReLU, Sequential
from transformers import BertForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

logger = logging.getLogger("cptlms")

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

        self.reparam_type = reparam_type  # type: ignore[unresolved-attribute]
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

    def forward(self) -> Annotated[Tensor, "virtual token"]:
        device = self.embedding.weight.device
        prompt_tokens = self.prompt_tokens.to(device)

        embeds = self.embedding(prompt_tokens).unsqueeze(0)
        if self.reparam_type == "lstm":
            embeds = self.mlp_head(self.lstm_head(embeds)[0])
        elif self.reparam_type == "mlp":
            embeds = self.mlp_head(embeds)

        return embeds


class PTuningBertQuestionAnswering(Module):
    def __init__(
        self,
        bert: BertForQuestionAnswering,
        num_virtual_tokens: int,
        train_new_layers: bool,
        encoder_hidden_size: int,
        encoder_reparam_type: EncoderReparameterizationType,
    ) -> None:
        super().__init__()

        self.num_virtual_tokens = num_virtual_tokens  # type: ignore[unresolved-attribute]

        self.bert = bert
        _freeze_params(self.bert, train_new_layers)

        bert_embedding = bert.get_input_embeddings()
        assert isinstance(bert_embedding, Embedding)

        self.bert_embedding = bert_embedding
        self.prompt_encoder = PromptEncoder(
            token_dim=bert_embedding.embedding_dim,
            num_virtual_tokens=num_virtual_tokens,
            hidden_size=encoder_hidden_size,
            reparam_type=encoder_reparam_type,
        )

    def forward(
        self,
        input_ids: Annotated[Tensor, "batch seq"],
        attention_mask: Annotated[Tensor, "batch seq"],
        start_positions: Annotated[Tensor, "batch"] | None = None,
        end_positions: Annotated[Tensor, "batch"] | None = None,
    ) -> QuestionAnsweringModelOutput:
        batch_size = input_ids.size(0)

        virtual_embeds: Annotated[Tensor, "batch virtual token"] = (
            self.prompt_encoder().expand(batch_size, -1, -1)
        )

        bert_embeds: Annotated[Tensor, "batch seq token"] = self.bert_embedding(
            input_ids
        )

        virtual_attention: Annotated[Tensor, "batch virtual"] = torch.ones(
            batch_size,
            self.num_virtual_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        out = self.bert(
            inputs_embeds=torch.cat([virtual_embeds, bert_embeds], dim=1),
            attention_mask=torch.cat([virtual_attention, attention_mask], dim=1),
            start_positions=start_positions,
            end_positions=end_positions,
        )

        assert isinstance(out, QuestionAnsweringModelOutput)

        start_logits = out.start_logits
        end_logits = out.end_logits

        assert start_logits is not None
        assert end_logits is not None

        out.start_logits = cast(FloatTensor, start_logits[:, self.num_virtual_tokens :])
        out.end_logits = cast(FloatTensor, end_logits[:, self.num_virtual_tokens :])

        return out


def _freeze_params(
    bert: BertForQuestionAnswering,
    train_new_layers: bool = True,
):
    logger.info("freeze bert parameters")
    for name, param in bert.named_parameters():
        if train_new_layers and "qa_outputs" in name:
            logger.info("skip %s", name)
            continue

        param.requires_grad = False
