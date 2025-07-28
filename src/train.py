from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        
        # Setup datamodule first 
        datamodule.setup()
        
        # Auto-resume support: set sampler position if resuming from checkpoint
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            import torch
            log.info(f"Resuming from checkpoint: {ckpt_path}")
            
            # Load checkpoint to get step count
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            completed_steps = checkpoint.get("global_step", 0)
            
            # CRITICAL: Restore max_steps for Lightning progress bar (Lightning loses this on resume)
            # NOTE: trainer.max_steps is read-only in newer Lightning versions, skip this fix
            # if cfg.trainer.get("max_steps"):
            #     trainer.max_steps = cfg.trainer.max_steps
            #     log.info(f"Restored trainer.max_steps = {trainer.max_steps} for progress bar")
            
            # CRITICAL: Fix Lightning's epoch jumping bug on resume
            # Lightning incorrectly increments current_epoch when resuming with StatefulDataLoader
            # For step-based training (max_steps), we want to stay in epoch 0
            # NOTE: Temporarily disabled to test dataset_length fix first
            # log.info(f"Before epoch fix: trainer.current_epoch = {trainer.current_epoch}")
            # trainer.current_epoch = 0
            # trainer.fit_loop.epoch_loop.batch_progress.current.completed = completed_steps
            # log.info(f"After epoch fix: trainer.current_epoch = {trainer.current_epoch} (corrected for step-based training)")
            
            # CRITICAL: Create train_dataloader to initialize sampler
            train_loader = datamodule.train_dataloader()
            
            # Set sampler position for exact resume (NEW: sampler-based approach)
            if hasattr(datamodule, 'train_sampler'):
                samples_per_step = cfg.data.batch_size * cfg.trainer.devices * cfg.trainer.accumulate_grad_batches
                total_samples_processed = completed_steps * samples_per_step
                
                # Set global start position in sampler
                datamodule.train_sampler.set_global_start(total_samples_processed)
                
                # Calculate coverage for logging
                if hasattr(datamodule, 'full_dataset'):
                    total_length = datamodule.full_dataset.total_dataset_length
                else:
                    total_length = len(datamodule.train_dataset.dataset) if hasattr(datamodule.train_dataset, 'dataset') else len(datamodule.train_dataset)
                
                coverage_pct = (total_samples_processed / total_length) * 100
                
                log.info(f"Resume calculation: completed_steps={completed_steps}, "
                        f"samples_per_step={samples_per_step}, total_samples_processed={total_samples_processed}")
                log.info(f"Dataset coverage: {coverage_pct:.3f}% ({total_samples_processed}/{total_length})")
                log.info(f"Sampler global start position: {total_samples_processed % total_length}")
            else:
                log.warning("No train_sampler found - resume may not be exact")
        
        # DEBUG: Check validation interval at runtime
        log.info(f"DEBUG: trainer.val_check_interval = {trainer.val_check_interval}")
        log.info(f"DEBUG: trainer.check_val_every_n_epoch = {trainer.check_val_every_n_epoch}")
        
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
