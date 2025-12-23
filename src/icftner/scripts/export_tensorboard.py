import polars as pl
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(logdir: str, outfile: str):
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()

    rows = []
    for tag in ea.Tags()["scalars"]:
        for event in ea.Scalars(tag):
            split, metric = tag.split("/")
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "step": event.step,
                    "value": event.value,
                }
            )

    df = pl.DataFrame(rows)
    df.write_csv(outfile)
