import logging

import typer

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def fine_tune(
    pretrained_model: str = "jhu-clsp/mmBERT-small",
    out_dir: str = "out/mmbert-small-ft",
    epochs: int = 10,
    batch_size: int = 32,
):
    from pathlib import Path

    import torch
    from transformers import (
        ModernBertForQuestionAnswering,
        PreTrainedTokenizerFast,
    )

    from cptlms.squad import Squad
    from cptlms.trainer import Trainer

    torch.set_float32_matmul_precision("high")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model)
    model = ModernBertForQuestionAnswering.from_pretrained(pretrained_model)
    squad = Squad(tokenizer)
    trainer = Trainer(
        model=model,
        epochs=epochs,
        qa_dataset=squad,
        collate_fn=Squad.default_collate_fn,
        batch_size=batch_size,
        out_dir=Path(out_dir),
    )

    trainer.train()


@app.command()
def p_tune():
    raise NotImplementedError()


if __name__ == "__main__":
    app()
