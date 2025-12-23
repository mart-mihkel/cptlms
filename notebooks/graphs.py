import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    from icftner.scripts.export_tensorboard import main as export_tensorboard
    from plotnine import (
        ggplot,
        aes,
        geom_line,
        theme_bw,
        facet_wrap,
        labs,
        ylim,
    )


@app.cell
def _():
    _file_ft = "out/ft.csv"
    _file_pt = "out/ot.csv"

    export_tensorboard(
        logdir="out/ft-multinerd-distilbert-base/tensorboard",
        outfile=_file_ft,
    )

    export_tensorboard(
        logdir="out/pt-multinerd-32v-mlp-distilbert-base//tensorboard",
        outfile=_file_pt,
    )

    _ft = pl.read_csv(_file_ft).with_columns(pl.lit("fine-tuning").alias("method"))
    _pt = pl.read_csv(_file_pt).with_columns(pl.lit("p-tuning").alias("method"))

    df = pl.concat([_ft, _pt])
    df
    return (df,)


@app.cell
def _(df):
    _df = df.filter(metric="loss")

    _p = (
        ggplot(_df)
        + aes("step", "value", color="split")
        + facet_wrap("method")
        + geom_line()
        + labs(
            title="DistilBERT base MultiNERD sequence classification",
            x="Step",
            y="Loss",
            color="Split",
        )
        + theme_bw()
    )

    _p.save("out/loss.png")
    _p
    return


@app.cell
def _(df):
    _df = df.filter(pl.col("metric").is_in(["accuracy", "precision", "recall"]))
    _df = _df.with_columns((_df["step"] / 1_000_000).alias("step_m"))

    _facet_labels = {
        "fine-tuning": "Fine-tuning",
        "prefix-tuning": "Prefix-tuning",
    }

    _color_labels = {"loss": "Loss", "f1": "F1 Score", "acc": "Accuracy"}

    _p = (
        ggplot(_df)
        + aes("step_m", "value", color="metric")
        + facet_wrap("method")
        + geom_line()
        + ylim(0.5, 1)
        + labs(
            x="Step (millions)",
            y="",
            color="Metric",
        )
        + theme_bw()
    )

    _p.save("out/metrics.png")
    _p
    return


if __name__ == "__main__":
    app.run()
