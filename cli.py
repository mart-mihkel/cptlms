import typer

app = typer.Typer()


@app.command()
def bert(
    pretrained_model_name: str = "bert-base-cased",
    trainer_output_dir: str = "out/bert",
    num_epochs: int = 3,
    batch_size: int = 16,
):
    from tuning.bert import main

    main(
        pretrained_model_name,
        trainer_output_dir,
        num_epochs,
        batch_size,
    )


if __name__ == "__main__":
    app()
