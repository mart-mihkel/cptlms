import json
import logging
import os
from pathlib import Path
from typing import Callable, TypedDict

import torch
from accelerate import Accelerator
from torch.nn.modules.module import Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from cptlms.squad import Squad, SquadMetrics

logger = logging.getLogger("cptlms")


class Telemetry(TypedDict):
    epoch: int
    train_loss: float
    eval_f1: float
    eval_exact_match: float


class Trainer:
    def __init__(
        self,
        model: Module,
        epochs: int,
        qa_dataset: Squad,
        collate_fn: Callable,
        batch_size: int = 16,
        out_dir: Path = Path("out/qa"),
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = qa_dataset
        self.out_dir = out_dir
        self.telemetry: list[Telemetry] = []

        model = model.to(self.device)

        train_loader = DataLoader(
            qa_dataset.train_tok.with_format("torch", device=self.device),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            qa_dataset.val_tok.with_format("torch", device=self.device),
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        optimizer = AdamW(
            model.parameters(),
            lr=2e-5,
            weight_decay=0.01,
        )

        self.accelerator = Accelerator(mixed_precision="fp16")
        self.model, self.optimizer, self.train_loader, self.val_loader = (
            self.accelerator.prepare(
                model,
                optimizer,
                train_loader,
                val_loader,
            )
        )

        self.model: Module
        self.optimizer: AdamW
        self.train_loader: DataLoader
        self.val_loader: DataLoader

        self.scheduler: LambdaLR = get_scheduler(
            "linear",
            num_warmup_steps=0,
            optimizer=self.optimizer,
            num_training_steps=len(self.train_loader) * self.epochs,
        )

    def train(self):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            logger.info("epoch %d/%d", epoch, self.epochs)
            self._epoch(epoch=epoch)
            self._save_checkpoint(epoch=epoch)
            self._save_telemetry()

    def _epoch(self, epoch: int):
        train_loss = self._train()
        eval_metrics = self._eval()

        logger.info("train loss: %.4f", train_loss)
        logger.info("eval em:    %.4f", eval_metrics["exact_match"])
        logger.info("eval f1:    %.4f", eval_metrics["f1"])

        self.telemetry.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_exact_match": eval_metrics["exact_match"],
                "eval_f1": eval_metrics["f1"],
            }
        )

    def _train(self) -> float:
        self.model.train()

        train_loss = 0
        pbar = tqdm(self.train_loader, desc="Train")
        for i, batch in enumerate(pbar):
            outputs = self.model(**batch)
            loss = outputs.loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_description(f"Train loss {train_loss / (i + 1):.4f}")

        return train_loss / len(self.train_loader)

    def _eval(self) -> SquadMetrics:
        self.model.eval()

        start_logits = []
        end_logits = []
        for batch in tqdm(self.val_loader, desc="Eval"):
            with torch.no_grad():
                outputs: QuestionAnsweringModelOutput = self.model(**batch)

            out_starts = self.accelerator.gather(outputs.start_logits)
            out_end = self.accelerator.gather(outputs.end_logits)

            assert isinstance(out_starts, torch.Tensor)
            assert isinstance(out_end, torch.Tensor)

            start_logits.append(out_starts.cpu())
            end_logits.append(out_end.cpu())

        start_logits = torch.cat(start_logits)
        end_logits = torch.cat(end_logits)

        n = len(self.val_loader) * self.batch_size
        start_logits = start_logits[:n]
        end_logits = end_logits[:n]

        return self.dataset.compute_metrics(start_logits, end_logits)

    def _save_checkpoint(self, epoch: int):
        os.makedirs(self.out_dir, exist_ok=True)
        save_path = self.out_dir / f"chkpt-{epoch}.pth"
        logger.info("save model state dict to %s", save_path)
        torch.save(self.model.state_dict(), save_path)

    def _save_telemetry(self):
        os.makedirs(self.out_dir, exist_ok=True)
        telem_path = self.out_dir / "telemetry.json"
        logger.info("save telemetry to %s", telem_path)
        with open(telem_path, "w") as f:
            json.dump(self.telemetry, f)
