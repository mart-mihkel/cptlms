import logging
from typing import Annotated, cast

import torch
from torch import FloatTensor, Tensor
from torch.nn import Embedding, Module
from torch.nn.functional import cross_entropy
from transformers import BertForQuestionAnswering
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from cptlms.models.prompt_encoder import EncoderReparameterizationType, PromptEncoder

logger = logging.getLogger("cptlms")


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

        skip_params = []
        if train_new_layers:
            skip_params = [
                "qa_outputs.bias",
                "qa_outputs.weight",
            ]

        _freeze_params(self.bert, skip_params=skip_params)

        bert_embedding = bert.get_input_embeddings()
        assert isinstance(bert_embedding, Embedding)
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

        bert_embedding = self.bert.get_input_embeddings()
        assert isinstance(bert_embedding, Embedding)
        bert_embeds: Annotated[Tensor, "batch seq token"] = bert_embedding(input_ids)

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


class PTuningBertSequenceClassification(Module):
    def __init__(
        self,
        bert: BertForSequenceClassification,
        num_virtual_tokens: int,
        train_new_layers: bool,
        encoder_hidden_size: int,
        encoder_reparam_type: EncoderReparameterizationType,
    ) -> None:
        super().__init__()

        self.num_virtual_tokens = num_virtual_tokens  # type: ignore[unresolved-attribute]

        self.bert = bert

        skip_params = []
        if train_new_layers:
            skip_params = [
                "classifier.bias",
                "classifier.weight",
                "pre_classifier.bias",
                "pre_classifier.weight",
            ]

        _freeze_params(self.bert, skip_params=skip_params)

        bert_embedding = bert.get_input_embeddings()
        assert isinstance(bert_embedding, Embedding)
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
        labels: Annotated[Tensor, "batch"],
    ) -> SequenceClassifierOutput:
        batch_size = input_ids.size(0)

        virtual_embeds: Annotated[Tensor, "batch virtual token"] = (
            self.prompt_encoder().expand(batch_size, -1, -1)
        )

        bert_embedding = self.bert.get_input_embeddings()
        assert isinstance(bert_embedding, Embedding)
        bert_embeds: Annotated[Tensor, "batch seq token"] = bert_embedding(input_ids)

        virtual_attention: Annotated[Tensor, "batch virtual"] = torch.ones(
            batch_size,
            self.num_virtual_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        out = self.bert(
            inputs_embeds=torch.cat([virtual_embeds, bert_embeds], dim=1),
            attention_mask=torch.cat([virtual_attention, attention_mask], dim=1),
        )

        assert isinstance(out, SequenceClassifierOutput)

        out_logits = out.logits
        assert isinstance(out_logits, Tensor)

        out["loss"] = cast(FloatTensor, cross_entropy(out_logits, labels))
        return out


def _freeze_params(model: Module, skip_params: list[str]):
    logger.info("freeze bert parameters")
    for name, param in model.named_parameters():
        if name in skip_params:
            logger.info("skip %s", name)
            continue

        param.requires_grad = False
