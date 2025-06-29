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

try:
    import wandb
except ImportError:
    wandb = None


class T5LitModule(LightningModule):
    """Lightning wrapper around :class:`~src.models.components.t5_model.T5Model`."""

    def __init__(
        self,
        t5_model: T5Model,
        optimizer: torch.optim.Optimizer,
        scheduler: Dict[str, Any] | None = None,
        tokenizer=None,  # Add tokenizer for span corruption logging
    ) -> None:  # noqa: D401
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = t5_model
        self.tokenizer = tokenizer

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, **batch):  # type: ignore[override]
        return self.model(**batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):  # type: ignore[override]
        outputs = self.forward(**batch)
        loss = outputs.loss

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", torch.exp(self.train_loss.compute()), prog_bar=True, on_epoch=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/learning_rate", current_lr, on_step=True, prog_bar=True)
        
        batch_size = batch["input_ids"].size(0)
        attention_mask = batch["attention_mask"]
        real_tokens = attention_mask.sum().item()
        total_tokens = attention_mask.numel()
        padding_ratio = 1 - (real_tokens / total_tokens)
        
        self.log("train/batch_size", batch_size, on_step=True)
        self.log("train/padding_ratio", padding_ratio, on_step=True)
        self.log("train/tokens_per_batch", real_tokens, on_step=True)
        
        if batch_idx % 100 == 0 and self.tokenizer is not None:
            self._log_span_corruption_examples(batch, outputs)
        
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
    
    def on_before_optimizer_step(self, optimizer):
        total_norm = 0
        param_count = 0
        
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        self.log("train/grad_norm", total_norm, on_step=True)
        self.log("train/trainable_params", param_count, on_step=True)
    
    def _log_span_corruption_examples(self, batch: Dict[str, torch.Tensor], outputs):
        if wandb is None or not hasattr(self.logger, 'experiment'):
            return
            
        try:
            input_ids = batch["input_ids"][0]
            labels = batch["labels"][0]
            
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            target_text = self.tokenizer.decode(labels, skip_special_tokens=False)
            
            with torch.no_grad():
                pred_ids = self.model.generate(
                    input_ids.unsqueeze(0),
                    max_length=50,
                    do_sample=False,
                    num_beams=1
                )
            pred_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=False)
            
            example_table = wandb.Table(
                columns=["Step", "Input", "Target", "Prediction"],
                data=[[self.global_step, input_text[:200], target_text[:200], pred_text[:200]]]
            )
            
            self.logger.experiment.log({
                "span_corruption_examples": example_table,
                "global_step": self.global_step
            })
            
        except Exception as e:
            pass 