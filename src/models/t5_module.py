from __future__ import annotations

"""
pytorch lightning module for t5 pre-training with span corruption.

this module is kept simple, offloading most work to hugging face's implementation.
it primarily handles:

logging metrics (loss + perplexity)
setting up optimizers and schedulers via hydra
leveraging lightning's support for mixed precision and distributed training.
"""

from typing import Any, Dict

import torch
from lightning import LightningModule
from torch.optim.optimizer import Optimizer
from torchmetrics import MeanMetric

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from .components.t5_model import T5Model


class T5LitModule(LightningModule):
    """Lightning wrapper around :class:`~src.models.components.t5_model.T5Model`."""

    def __init__(
        self,
        t5_model: T5Model,
        optimizer: torch.optim.Optimizer,
        scheduler: Dict[str, Any] | None = None,
    ) -> None:  # noqa: D401
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = t5_model

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, **batch):  # type: ignore[override]
        return self.model(**batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):  # type: ignore[override]
        outputs = self.forward(**batch)
        loss = outputs.loss

        # Metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", torch.exp(self.train_loss.compute()), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):  # type: ignore[override]
        outputs = self.forward(**batch)
        loss = outputs.loss

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", torch.exp(self.val_loss.compute()), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):  # type: ignore[override]
        optimizer_cfg = self.hparams.optimizer  # comes from Hydra instantiation
        if isinstance(optimizer_cfg, Optimizer):
            optimizer = optimizer_cfg
        else:
            # hydra passes instantiated object, but if mis-configured fallback
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        if self.hparams.scheduler is None:
            return optimizer

        scheduler_cfg = self.hparams.scheduler
        if isinstance(scheduler_cfg, dict):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_cfg.get("num_warmup_steps", 1000),
                num_training_steps=scheduler_cfg.get("num_training_steps", 100_000),
                num_cycles=scheduler_cfg.get("num_cycles", 1),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer 