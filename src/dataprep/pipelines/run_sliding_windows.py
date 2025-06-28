

import os
import sys
import logging
from datetime import datetime


script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, proj_root)

import hydra
from omegaconf import DictConfig, OmegaConf
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
# from datatrove.pipeline.stats import DocStats  # Not needed for sliding window preprocessing

# Import sliding window processor
from src.dataprep.sliding_window_processor import SlidingWindowProcessor

# Setup logging
log = logging.getLogger(__name__)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not available - sliding window preprocessing analytics will not be logged")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="preprocessing/sliding_windows")
def main(cfg: DictConfig) -> None:
    log.info("Starting T5 SentencePiece Token Count Preprocessing Pipeline")
    log.info("Configuration:")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # W&B Session initialisieren
    wandb_session = None
    if WANDB_AVAILABLE and cfg.sliding_windows.get("log_to_wandb", False):
        try:
            # Generate automatic tags based on configuration
            tags = ["sliding-windows", "preprocessing", "tokenization"]
            
            if cfg.sliding_windows.limit_documents != -1:
                tags.extend(["test_run", f"limit-{cfg.sliding_windows.limit_documents}"])
            else:
                tags.append("full-dataset")
            
            wandb_session = wandb.init(
                project=cfg.sliding_windows.wandb.project,
                group=cfg.sliding_windows.wandb.group,
                tags=tags,
                job_type="sliding-window-preprocessing",
                notes="Precompute T5 SentencePiece token counts for on-the-fly sliding windows",
                config=OmegaConf.to_container(cfg, resolve=True)
            )
            log.info(f"W&B session initialized: {wandb_session.url}")
        except Exception as e:
            log.warning(f"Failed to initialize W&B session: {e}")
            wandb_session = None

    pipeline = [
        ParquetReader(
            data_folder=cfg.sliding_windows.paths.input_folder,
            glob_pattern=cfg.sliding_windows.paths.input_pattern,
            limit=cfg.sliding_windows.limit_documents,
            text_key=cfg.sliding_windows.reader.text_key,
            id_key=cfg.sliding_windows.reader.id_key,
            default_metadata=OmegaConf.to_container(cfg.sliding_windows.reader.default_metadata, resolve=True)
        ),
        
        SlidingWindowProcessor(
            tokenizer_name_or_path=cfg.sliding_windows.tokenizer.name_or_path,
            max_length=cfg.sliding_windows.window_config.max_length,
            overlap_size=cfg.sliding_windows.window_config.overlap_size,
            log_to_wandb=cfg.sliding_windows.get("log_to_wandb", False) and bool(wandb_session),
            wandb_project=cfg.sliding_windows.wandb.project,
            wandb_group=cfg.sliding_windows.wandb.group
        ),
        
        ParquetWriter(cfg.sliding_windows.paths.output_folder)
    ]

    log.info(f"Pipeline: {len(pipeline)} steps")
    log.info(f"Tasks: {cfg.sliding_windows.execution.tasks}, Workers: {cfg.sliding_windows.execution.workers}")
    log.info(f"Max Length: {cfg.sliding_windows.window_config.max_length}, Overlap: {cfg.sliding_windows.window_config.overlap_size}")
    log.info(f"Tokenizer: {cfg.sliding_windows.tokenizer.name_or_path}")
    
    if cfg.sliding_windows.limit_documents != -1:
        log.info(f"Test mode: {cfg.sliding_windows.limit_documents} docs")
    else:
        log.info("Full processing mode")
    
    log.info(f"Input: {cfg.sliding_windows.paths.input_folder}")
    log.info(f"Output: {cfg.sliding_windows.paths.output_folder}")
    
    if wandb_session:
        log.info(f"W&B: {wandb_session.name}")
    else:
        log.info("W&B: Disabled")

    # Execute pipeline
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.sliding_windows.execution.tasks,
        workers=cfg.sliding_windows.execution.workers,
        logging_dir=cfg.sliding_windows.paths.get("logging_dir", "logs/sliding_windows")
    )
    
    try:
        log.info("Starting pipeline...")
        start_time = datetime.now()
        
        executor.run()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        log.info("Pipeline completed!")
        log.info(f"Time: {execution_time:.1f}s")
        
        # Log execution metrics to W&B
        if wandb_session:
            wandb.log({
                "pipeline_execution_time": execution_time,
                "limit_documents": cfg.sliding_windows.limit_documents,
                "max_length": cfg.sliding_windows.window_config.max_length,
                "overlap_size": cfg.sliding_windows.window_config.overlap_size,
                "tasks": cfg.sliding_windows.execution.tasks,
                "workers": cfg.sliding_windows.execution.workers,
                "tokenizer": cfg.sliding_windows.tokenizer.name_or_path
            })
        
        # W&B Session beenden
        if wandb_session:
            try:
                wandb.finish()
                log.info("W&B session finished")
            except Exception as e:
                log.warning(f"Error finishing W&B session: {e}")
            
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        # W&B Session auch bei Fehler beenden
        if wandb_session:
            try:
                wandb.finish()
            except:
                pass
        raise


if __name__ == "__main__":
    main() 