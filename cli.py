import json
import logging
import os
import sys
from logging import FileHandler, StreamHandler
from pathlib import Path
from typing import Any, Literal

from typer import Context, Typer

app = Typer(add_completion=False)
logger = logging.getLogger(__name__)


def _setup_logging(out_dir: str):
    out_path = Path(out_dir)
    log_path = out_path / "logs.log"

    os.makedirs(out_path, exist_ok=True)
    handlers = [StreamHandler(sys.stdout), FileHandler(log_path)]
    logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(message)s")
    logger.info("set logger file handler to %s", log_path)


def _save_params(out_dir: str, params: dict[str, Any]):
    out_path = Path(out_dir)
    params_path = out_path / "cli-params.json"
    os.makedirs(out_path, exist_ok=True)

    logger.info("save cli input params to %s", params_path)
    with open(params_path, "w") as f:
        json.dump(params, f)


@app.command(help="Export tensorboard scalars to csv")
def export_tensorboard(logdir: str = "out", outfile: str = "tb.csv"):
    from icftner.scripts.export_tensorboard import main

    main(logdir=logdir, outfile=outfile)


@app.command(
    help="Fine tune a pretrained bert model for question answering on SQuAD dataset"
)
def ft_bert_squad(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/ft-squad",
    epochs: int = 20,
    head_only: bool = False,
    train_split: str = "train",
    eval_split: str = "validation",
):
    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    from icftner.scripts.ft_bert_squad import main

    main(
        pretrained_model=pretrained_model,
        out_dir=out_dir,
        epochs=epochs,
        head_only=head_only,
        train_split=train_split,
        eval_split=eval_split,
    )


@app.command(
    help="P-tune a pretrained bert model for question answering on SQuAD dataset"
)
def pt_bert_squad(
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
    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    from icftner.scripts.pt_bert_squad import main

    main(
        pretrained_model=pretrained_model,
        out_dir=out_dir,
        epochs=epochs,
        num_virtual_tokens=num_virtual_tokens,
        train_new_layers=train_new_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparam_type=encoder_reparam_type,
        train_split=train_split,
        eval_split=eval_split,
    )


@app.command(
    help="Benchark a pretrained bert model for sequence classification on MultiNERD dataset"
)
def benchmark_bert_multinerd(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/benchmark-multinerd",
    english_only: bool = True,
    test_split: str = "test",
):
    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    from icftner.scripts.benchmark_bert_multinerd import main

    main(
        pretrained_model=pretrained_model,
        out_dir=out_dir,
        english_only=english_only,
        test_split=test_split,
    )


@app.command(
    help="Fine tune a pretrained bert model for sequence classification on MultiNERD dataset"
)
def ft_bert_multinerd(
    ctx: Context,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/ft-multinerd",
    epochs: int = 5,
    head_only: bool = False,
    english_only: bool = True,
    train_split: str = "train",
    eval_split: str = "validation",
):
    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    from icftner.scripts.ft_bert_multinerd import main

    main(
        pretrained_model=pretrained_model,
        out_dir=out_dir,
        epochs=epochs,
        head_only=head_only,
        english_only=english_only,
        train_split=train_split,
        eval_split=eval_split,
    )


@app.command(
    help="P-tune a pretrained bert model for sequence classification on MultiNERD dataset"
)
def pt_bert_multinerd(
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
    _setup_logging(out_dir=out_dir)
    _save_params(out_dir=out_dir, params=ctx.params)

    from icftner.scripts.pt_bert_multinerd import main

    main(
        pretrained_model=pretrained_model,
        out_dir=out_dir,
        epochs=epochs,
        num_virtual_tokens=num_virtual_tokens,
        train_new_layers=train_new_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparam_type=encoder_reparam_type,
        english_only=english_only,
        train_split=train_split,
        eval_split=eval_split,
    )


if __name__ == "__main__":
    app()
