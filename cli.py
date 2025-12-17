import json
import logging
import os
import sys
from logging import FileHandler, StreamHandler
from typing import Any, Literal

from typer import Context, Typer


app = Typer(add_completion=False)
logger = logging.getLogger("cptlms")


def _setup_logging(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{out_dir}/logs.log"
    log_fmt = "%(message)s"
    handlers = [StreamHandler(sys.stdout), FileHandler(log_path)]
    logging.basicConfig(level=logging.INFO, handlers=handlers, format=log_fmt)
    logger.info("set logger file handler to %s", log_path)


def _save_params(out_dir: str, params: dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    params_path = f"{out_dir}/cli-params.json"
    logger.info("save cli input params to %s", params_path)
    with open(params_path, "w") as f:
        json.dump(params, f)


@app.command(
    help="Fine tune a pretrained bert model for question answering on SQuAD dataset"
)
def finetune_bert_squad(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/ft-squad",
    epochs: int = 20,
    train_split: str = "train",
    eval_split: str = "validation",
):
    from datasets.arrow_dataset import Dataset
    from datasets.load import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments

    from cptlms.datasets.squad import squad_collate_fn, tokenize_squad

    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    logger.info("load squad")
    train, eval = load_dataset("squad", split=[train_split, eval_split])
    assert isinstance(train, Dataset)
    assert isinstance(eval, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    logger.info("tokenize squad")
    train_tokenized = tokenize_squad(tokenizer, train)
    eval_tokenized = tokenize_squad(tokenizer, eval)

    logger.info("load %s", pretrained_model)
    bert = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)

    logger.info("init trainer")
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=f"{out_dir}/tensorboard",
        logging_steps=2500,
        report_to="tensorboard",
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="epoch",
        auto_find_batch_size=True,
        fp16=True,
    )

    trainer = Trainer(
        bert,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=squad_collate_fn,
    )

    trainer.train()


@app.command(
    help="P-tune a pretrained bert model for question answering on SQuAD dataset"
)
def ptune_bert_squad(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/pt-squad",
    epochs: int = 5,
    num_virtual_tokens: int = 32,
    train_new_layers: bool = True,
    encoder_hidden_size: int = 128,
    encoder_reparam_type: Literal["emb", "mlp", "lstm"] = "mlp",
    train_split: str = "train",
    eval_split: str = "validation",
):
    from datasets.arrow_dataset import Dataset
    from datasets.load import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments

    from cptlms.datasets.squad import squad_collate_fn, tokenize_squad
    from cptlms.models.bert import PTuningBertQuestionAnswering

    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    logger.info("load squad")
    train, eval = load_dataset("squad", split=[train_split, eval_split])
    assert isinstance(train, Dataset)
    assert isinstance(eval, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    logger.info("tokenize squad")
    train_tokenized = tokenize_squad(tokenizer, train)
    eval_tokenized = tokenize_squad(tokenizer, eval)

    logger.info("load %s", pretrained_model)
    bert = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)

    logger.info("init ptunging %s", pretrained_model)
    pt_bert = PTuningBertQuestionAnswering(
        bert=bert,
        num_virtual_tokens=num_virtual_tokens,
        train_new_layers=train_new_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparam_type=encoder_reparam_type,
    )

    logger.info("init trainer")
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=f"{out_dir}/tensorboard",
        logging_steps=2500,
        report_to="tensorboard",
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="epoch",
        auto_find_batch_size=True,
        fp16=True,
    )

    trainer = Trainer(
        pt_bert,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=squad_collate_fn,
    )

    trainer.train()


@app.command(
    help="Benchark a pretrained bert model for sequence classification on MultiNERD dataset"
)
def benchmark_bert_multinerd(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/benchmark-multinerd",
    english_only: bool = True,
    eval_split: str = "validation",
):
    from datasets.arrow_dataset import Dataset
    from datasets.load import load_dataset
    from datasets.utils.info_utils import VerificationMode
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
    )
    from transformers.models.auto.modeling_auto import (
        AutoModelForSequenceClassification,
    )
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments

    from cptlms.datasets.multinerd import (
        MULTINERD_ID2TAG,
        MULTINERD_TAG2ID,
        filter_multinerd_english,
        tokenize_multinerd_prompted,
    )
    from cptlms.datasets.multinerd import compute_multinerd_prompted_metrics

    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    logger.info("load multinerd")
    eval = load_dataset(
        "Babelscape/multinerd",
        split=eval_split,
        verification_mode=VerificationMode.NO_CHECKS,
    )

    assert isinstance(eval, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if english_only:
        logger.info("filter multinerd english")
        eval = eval.filter(filter_multinerd_english, batched=True)

    logger.info("tokenize multinerd prompted")
    eval_tokenized = tokenize_multinerd_prompted(
        tokenizer=tokenizer,
        data=eval,
        with_system_prompt=False,
    )

    logger.info("load %s", pretrained_model)
    bert = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(MULTINERD_ID2TAG),
        id2label=MULTINERD_ID2TAG,
        label2id=MULTINERD_TAG2ID,
    )

    logger.info("init trainer")
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=f"{out_dir}/tensorboard",
        auto_find_batch_size=True,
        fp16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        bert,
        args=args,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_multinerd_prompted_metrics,
    )

    trainer.evaluate()


@app.command(
    help="Fine tune a pretrained bert model for sequence classification on MultiNERD dataset"
)
def finetune_bert_multinerd(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/ft-multinerd",
    epochs: int = 5,
    english_only: bool = True,
    train_split: str = "train",
    eval_split: str = "validation",
):
    from datasets.arrow_dataset import Dataset
    from datasets.load import load_dataset
    from datasets.utils.info_utils import VerificationMode
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
    )
    from transformers.models.auto.modeling_auto import (
        AutoModelForSequenceClassification,
    )
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments

    from cptlms.datasets.multinerd import (
        MULTINERD_ID2TAG,
        MULTINERD_TAG2ID,
        filter_multinerd_english,
        tokenize_multinerd_prompted,
    )
    from cptlms.datasets.multinerd import compute_multinerd_prompted_metrics

    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    logger.info("load multinerd")
    train, eval = load_dataset(
        "Babelscape/multinerd",
        split=[train_split, eval_split],
        verification_mode=VerificationMode.NO_CHECKS,
    )

    assert isinstance(train, Dataset)
    assert isinstance(eval, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if english_only:
        logger.info("filter multinerd english")
        train = train.filter(filter_multinerd_english, batched=True)
        eval = eval.filter(filter_multinerd_english, batched=True)

    logger.info("tokenize multinerd prompted")
    train_tokenized = tokenize_multinerd_prompted(
        tokenizer=tokenizer,
        data=train,
        with_system_prompt=False,
    )

    eval_tokenized = tokenize_multinerd_prompted(
        tokenizer=tokenizer,
        data=eval,
        with_system_prompt=False,
    )

    logger.info("load %s", pretrained_model)
    bert = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(MULTINERD_ID2TAG),
        id2label=MULTINERD_ID2TAG,
        label2id=MULTINERD_TAG2ID,
    )

    logger.info("init trainer")
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=f"{out_dir}/tensorboard",
        logging_steps=5000,
        report_to="tensorboard",
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=10000,
        save_strategy="epoch",
        auto_find_batch_size=True,
        fp16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        bert,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_multinerd_prompted_metrics,
    )

    trainer.train()


@app.command(
    help="P-tune a pretrained bert model for sequence classification on MultiNERD dataset"
)
def ptune_bert_multinerd(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/pt-multinerd",
    epochs: int = 5,
    num_virtual_tokens: int = 32,
    train_new_layers: bool = True,
    encoder_hidden_size: int = 128,
    encoder_reparam_type: Literal["emb", "mlp", "lstm"] = "mlp",
    english_only: bool = True,
    train_split: str = "train",
    eval_split: str = "validation",
):
    from datasets.arrow_dataset import Dataset
    from datasets.load import load_dataset
    from datasets.utils.info_utils import VerificationMode
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
    )
    from transformers.models.auto.modeling_auto import (
        AutoModelForSequenceClassification,
    )
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments

    from cptlms.datasets.multinerd import (
        MULTINERD_ID2TAG,
        MULTINERD_TAG2ID,
        filter_multinerd_english,
        tokenize_multinerd_prompted,
    )
    from cptlms.models.bert import PTuningBertSequenceClassification
    from cptlms.datasets.multinerd import compute_multinerd_prompted_metrics

    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    logger.info("load multinerd")
    train, eval = load_dataset(
        "Babelscape/multinerd",
        split=[train_split, eval_split],
        verification_mode=VerificationMode.NO_CHECKS,
    )

    assert isinstance(train, Dataset)
    assert isinstance(eval, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if english_only:
        logger.info("filter multinerd english")
        train = train.filter(filter_multinerd_english, batched=True)
        eval = eval.filter(filter_multinerd_english, batched=True)

    logger.info("tokenize multinerd prompted")
    train_tokenized = tokenize_multinerd_prompted(
        tokenizer=tokenizer,
        data=train,
        with_system_prompt=False,
    )

    eval_tokenized = tokenize_multinerd_prompted(
        tokenizer=tokenizer,
        data=eval,
        with_system_prompt=False,
    )

    logger.info("load %s", pretrained_model)
    bert = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(MULTINERD_ID2TAG),
        id2label=MULTINERD_ID2TAG,
        label2id=MULTINERD_TAG2ID,
    )

    logger.info("init ptunging %s", pretrained_model)
    pt_bert = PTuningBertSequenceClassification(
        bert=bert,
        num_virtual_tokens=num_virtual_tokens,
        train_new_layers=train_new_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparam_type=encoder_reparam_type,
    )

    total_params = sum(p.numel() for p in pt_bert.parameters())
    trainable_params = sum(p.numel() for p in pt_bert.parameters() if p.requires_grad)
    logger.info("total params:     %d", total_params)
    logger.info("trainable params: %d", trainable_params)

    logger.info("init trainer")
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=f"{out_dir}/tensorboard",
        logging_steps=5000,
        report_to="tensorboard",
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=10000,
        save_strategy="epoch",
        auto_find_batch_size=True,
        fp16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        pt_bert,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_multinerd_prompted_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    app()
