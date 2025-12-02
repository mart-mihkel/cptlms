import logging
import os
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
import torch
from accelerate import Accelerator
from torch.nn.modules.module import Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from cptlms.squad import SQuAD, SQuADMetrics

logger = logging.getLogger(__name__)


class Telemetry(TypedDict):
    metrics: list[SQuADMetrics]
    train_loss: list[float]


class Trainer:
    def __init__(
        self,
        model: Module,
        epochs: int,
        qa_dataset: SQuAD,
        collate_fn: Callable,
        batch_size: int = 16,
        out_dir: Path = Path("out/qa"),
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = qa_dataset
        self.out_dir = out_dir
        self.telemetry: Telemetry = {"metrics": [], "train_loss": []}

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
            self._epoch()
            self._save_checkpoint(epoch=epoch)

    def _epoch(self):
        train_loss = 0

        for batch in tqdm(self.train_loader, desc="train"):
            outputs = self.model(**batch)
            loss = outputs.loss
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(self.train_loader)
        logger.info("training loss: %.4f", avg_train_loss)

        metrics = self._eval()
        logger.info("exact match: %.4f", metrics["exact_match"])
        logger.info("f1: %.4f", metrics["f1"])

        self.telemetry["train_loss"].append(avg_train_loss)
        self.telemetry["metrics"].append(metrics)

    def _eval(self) -> SQuADMetrics:
        self.model.eval()

        start_logits = []
        end_logits = []

        for batch in tqdm(self.val_loader, desc="eval"):
            with torch.no_grad():
                outputs: QuestionAnsweringModelOutput = self.model(**batch)

            out_starts = self.accelerator.gather(outputs.start_logits)
            out_end = self.accelerator.gather(outputs.end_logits)

            assert isinstance(out_starts, torch.Tensor)
            assert isinstance(out_end, torch.Tensor)

            start_logits.append(out_starts.cpu().numpy())
            end_logits.append(out_end.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)

        n = len(self.val_loader) * self.batch_size
        start_logits = start_logits[:n]
        end_logits = end_logits[:n]

        return self.dataset.compute_metrics(start_logits, end_logits)

    def _save_checkpoint(self, epoch: int):
        os.makedirs(self.out_dir, exist_ok=True)
        save_path = self.out_dir / f"chkpt-{epoch}.pth"
        logger.info("save model state dict to %s", save_path)
        torch.save(self.model.state_dict(), save_path)
