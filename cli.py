import json
import logging
import os
import sys
from logging import FileHandler, StreamHandler
from pathlib import Path
from typing import Any, Literal

from typer import Context, Typer

from cptlms.datasets.multinerd import MULTINERD_ID2TAG, MULTINERD_TAG2ID

app = Typer(add_completion=False)
logger = logging.getLogger("cptlms")


def _setup_logging(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{out_dir}/logs.log"
    handlers = [StreamHandler(sys.stdout), FileHandler(log_path)]
    logging.basicConfig(level=logging.INFO, handlers=handlers)
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
    batch_size: int = 32,
    train_split: str = "train",
    val_split: str = "validation",
):
    from datasets.arrow_dataset import Dataset
    from datasets.load import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from cptlms.datasets.squad import squad_collate_fn, tokenize_squad
    from cptlms.trainer import Trainer

    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    logger.info("load squad")
    train, val = load_dataset("squad", split=[train_split, val_split])
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    logger.info("tokenize squad")
    train_tokenized = tokenize_squad(tokenizer, train)
    val_tokenized = tokenize_squad(tokenizer, val)

    logger.info("load %s", pretrained_model)
    model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total parameters:     %d", total_params)
    logger.info("trainable parameters: %d", trainable_params)

    trainer = Trainer(
        model=model,
        epochs=epochs,
        train_data=train_tokenized.with_format("torch"),
        val_data=val_tokenized.with_format("torch"),
        batch_size=batch_size,
        collate_fn=squad_collate_fn,
        out_dir=Path(out_dir),
    )

    trainer.train()


@app.command(
    help="P-tune a pretrained bert model for question answering on SQuAD dataset"
)
def ptune_bert_squad(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/pt-squad",
    epochs: int = 20,
    batch_size: int = 32,
    num_virtual_tokens: int = 32,
    train_new_layers: bool = True,
    encoder_hidden_size: int = 128,
    encoder_reparam_type: Literal["emb", "mlp", "lstm"] = "mlp",
    train_split: str = "train",
    val_split: str = "validation",
):
    from datasets.arrow_dataset import Dataset
    from datasets.load import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from cptlms.datasets.squad import squad_collate_fn, tokenize_squad
    from cptlms.models.bert import PTuningBertQuestionAnswering
    from cptlms.trainer import Trainer

    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    logger.info("load squad")
    train, val = load_dataset("squad", split=[train_split, val_split])
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    logger.info("tokenize squad")
    train_tokenized = tokenize_squad(tokenizer, train)
    val_tokenized = tokenize_squad(tokenizer, val)

    logger.info("load %s", pretrained_model)
    model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)

    logger.info("init ptunging %s", pretrained_model)
    pt_bert = PTuningBertQuestionAnswering(
        bert=model,
        num_virtual_tokens=num_virtual_tokens,
        train_new_layers=train_new_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparam_type=encoder_reparam_type,
    )

    total_params = sum(p.numel() for p in pt_bert.parameters())
    trainable_params = sum(p.numel() for p in pt_bert.parameters() if p.requires_grad)
    logger.info("total params:     %d", total_params)
    logger.info("trainable params: %d", trainable_params)

    trainer = Trainer(
        model=pt_bert,
        epochs=epochs,
        train_data=train_tokenized.with_format("torch"),
        val_data=val_tokenized.with_format("torch"),
        batch_size=batch_size,
        collate_fn=squad_collate_fn,
        out_dir=Path(out_dir),
    )

    trainer.train()


@app.command(
    help="P-tune a pretrained bert model for sequence classification on MultiNERD dataset"
)
def ptune_bert_multinerd(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/pt-multinerd",
    epochs: int = 20,
    batch_size: int = 32,
    num_virtual_tokens: int = 32,
    train_new_layers: bool = True,
    encoder_hidden_size: int = 128,
    encoder_reparam_type: Literal["emb", "mlp", "lstm"] = "mlp",
    english_only: bool = True,
    train_split: str = "train",
    val_split: str = "validation",
    test_split: str = "test",
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

    from cptlms.datasets.multinerd import (
        filter_multinerd_english,
        tokenize_multinerd_prompted,
    )
    from cptlms.models.bert import PTuningBertSequenceClassification
    from cptlms.trainer import Trainer

    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    logger.info("load multinerd")
    train, val, _ = load_dataset(
        "Babelscape/multinerd",
        split=[train_split, val_split, test_split],
        verification_mode=VerificationMode.NO_CHECKS,
    )

    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if english_only:
        logger.info("filter multinerd english")
        train = train.filter(filter_multinerd_english, batched=True)
        val = val.filter(filter_multinerd_english, batched=True)

    logger.info("tokenize multinerd")
    multinerd_cache_path = Path("out/multinerd-tokenized")
    train_tokenized = tokenize_multinerd_prompted(
        tokenizer=tokenizer,
        data=train,
        cache_path=multinerd_cache_path / "train",
    )

    val_tokenized = tokenize_multinerd_prompted(
        tokenizer=tokenizer,
        data=val,
        cache_path=multinerd_cache_path / "validation",
    )

    logger.info("load %s", pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(MULTINERD_ID2TAG),
        id2label=MULTINERD_ID2TAG,
        label2id=MULTINERD_TAG2ID,
    )

    logger.info("init ptunging %s", pretrained_model)
    pt_bert = PTuningBertSequenceClassification(
        bert=model,
        num_virtual_tokens=num_virtual_tokens,
        train_new_layers=train_new_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparam_type=encoder_reparam_type,
    )

    total_params = sum(p.numel() for p in pt_bert.parameters())
    trainable_params = sum(p.numel() for p in pt_bert.parameters() if p.requires_grad)
    logger.info("total params:     %d", total_params)
    logger.info("trainable params: %d", trainable_params)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=pt_bert,
        epochs=epochs,
        train_data=train_tokenized.with_format("torch"),
        val_data=val_tokenized.with_format("torch"),
        batch_size=batch_size,
        collate_fn=data_collator,
        out_dir=Path(out_dir),
    )

    trainer.train()


if __name__ == "__main__":
    app()
