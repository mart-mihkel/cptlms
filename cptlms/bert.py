import logging
from typing import cast

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from .util.datasets import tokenize_squad

logger = logging.getLogger(__name__)


def bert_qa(
    pretrained_model_name: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
):
    logger.info("initializing pretrained model from %s", pretrained_model_name)

    model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name)

    logger.info("initializing pretrained tokenizer from %s", pretrained_model_name)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)

    logger.info("tokenizing SQuAD", pretrained_model_name)

    squad = load_dataset("squad")
    squad = cast(DatasetDict, squad)
    squad = tokenize_squad(squad, tokenizer)

    _train(
        model,
        tokenizer,
        squad,
        output_dir,
        num_epochs,
        batch_size,
    )


def _train(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetDict,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
):
    collator = DefaultDataCollator()

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir="logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    logger.info("start training")

    trainer.train()
