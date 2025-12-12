import logging
from typing import Literal

import typer

app = typer.Typer(add_completion=False)
logger = logging.getLogger("cptlms")


def _setup_logging(out_dir: str):
    import os
    import sys
    from logging import FileHandler, StreamHandler

    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{out_dir}/logs.log"
    handlers = [StreamHandler(sys.stdout), FileHandler(log_path)]
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logger.info("set logger file handler to %s", log_path)


def _save_params(out_dir: str, **kwargs):
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    params_path = f"{out_dir}/cli-params.json"
    logger.info("save cli input params to %s", params_path)
    with open(params_path, "w") as f:
        json.dump(kwargs, f)


@app.command(
    help="Fine tune a pretrained model for question answering on SQuAD dataset"
)
def fine_tune_qa(
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/ft",
    epochs: int = 20,
    batch_size: int = 32,
    train_split: str = "train",
    eval_split: str = "validation",
):
    from pathlib import Path

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from cptlms.squad import Squad
    from cptlms.trainer import Trainer

    _setup_logging(out_dir=out_dir)
    _save_params(
        epochs=epochs,
        out_dir=out_dir,
        batch_size=batch_size,
        eval_split=eval_split,
        train_split=train_split,
        pretrained_model=pretrained_model,
    )

    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    squad = Squad(tokenizer, train_split=train_split, eval_split=eval_split)

    model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total parameters:     %d", total_params)
    logger.info("trainable parameters: %d", trainable_params)

    trainer = Trainer(
        model=model,
        epochs=epochs,
        qa_dataset=squad,
        batch_size=batch_size,
        collate_fn=Squad.collate_fn,
        out_dir=Path(out_dir),
    )

    trainer.train()


@app.command(help="P-tune a pretrained model for question answering on SQuAD dataset")
def p_tune_qa(
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/pt",
    epochs: int = 20,
    batch_size: int = 32,
    num_virtual_tokens: int = 32,
    train_new_layers: bool = True,
    encoder_hidden_size: int = 128,
    encoder_reparam_type: Literal["emb", "mlp", "lstm"] = "mlp",
    train_split: str = "train",
    eval_split: str = "validation",
):
    from pathlib import Path

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from cptlms.ptuning import PTuningBertQuestionAnswering
    from cptlms.squad import Squad
    from cptlms.trainer import Trainer

    _setup_logging(out_dir=out_dir)
    _save_params(
        epochs=epochs,
        out_dir=out_dir,
        batch_size=batch_size,
        eval_split=eval_split,
        train_split=train_split,
        pretrained_model=pretrained_model,
        train_new_layers=train_new_layers,
        num_virtual_tokens=num_virtual_tokens,
        encoder_reparam_type=encoder_reparam_type,
    )

    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    squad = Squad(tokenizer, train_split=train_split, eval_split=eval_split)

    base_bert = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)
    pt_bert = PTuningBertQuestionAnswering(
        bert=base_bert,
        num_virtual_tokens=num_virtual_tokens,
        train_new_layers=train_new_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparam_type=encoder_reparam_type,
    )

    total_params = sum(p.numel() for p in pt_bert.parameters())
    trainable_params = sum(p.numel() for p in pt_bert.parameters() if p.requires_grad)
    logger.info("total parameters:     %d", total_params)
    logger.info("trainable parameters: %d", trainable_params)

    trainer = Trainer(
        model=pt_bert,
        epochs=epochs,
        qa_dataset=squad,
        batch_size=batch_size,
        collate_fn=Squad.collate_fn,
        out_dir=Path(out_dir),
    )

    trainer.train()


if __name__ == "__main__":
    app()
