from typing import cast

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    BertForMultipleChoice,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForMultipleChoice,
)

from .util.datasets import tokenize_swag


def main(
    pretrained_model_name: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
):
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
    tokenizer = cast(BertTokenizerFast, tokenizer)

    swag = load_dataset("swag", "regular")
    swag = cast(DatasetDict, swag)
    swag = tokenize_swag(swag, tokenizer)

    bert = BertForMultipleChoice.from_pretrained(pretrained_model_name)

    _train(
        bert,
        tokenizer,
        swag,
        output_dir,
        num_epochs,
        batch_size,
    )


def _train(
    model: torch.nn.Module,
    tokenizer: BertTokenizerFast,
    dataset: DatasetDict,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
):
    collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir="logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
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

    trainer.train()
