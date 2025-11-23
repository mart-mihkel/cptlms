import typer

app = typer.Typer()


@app.command()
def bert(
    pretrained_model_name: str = "jhu-clsp/mmBERT-small",
    trainer_output_dir: str = "out/bert",
    num_epochs: int = 3,
    batch_size: int = 16,
):
    from cptlms.bert import bert_qa

    bert_qa(
        pretrained_model_name,
        trainer_output_dir,
        num_epochs,
        batch_size,
    )


if __name__ == "__main__":
    app()
