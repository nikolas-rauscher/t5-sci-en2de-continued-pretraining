import hydra
from omegaconf import DictConfig, OmegaConf
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.filters import URLFilter, SamplerFilter
from datatrove.pipeline.stats import DocStats

import os, sys
# Ensure project root is on PYTHONPATH  for src package
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, proj_root)

from src.dataprep.cleaning_blocks import StructMetaToJSON

@hydra.main(version_base="1.3", config_path="../configs", config_name="cleaning/datatrove")
def main(cfg: DictConfig) -> None:


    pipeline = [
        ParquetReader(data_folder=cfg.cleaning.paths.src_dir, glob_pattern=cfg.cleaning.paths.src_pattern, limit=cfg.cleaning.cleaning.limit_documents),
        #URLFilter(),
        #StructMetaToJSON(),
        SamplerFilter(),
        DocStats(output_folder=cfg.cleaning.paths.dst + "/stats", groups_to_compute=["summary"]),
        ParquetWriter(cfg.cleaning.paths.dst),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.cleaning.cleaning.tasks,
        workers=cfg.cleaning.cleaning.workers,
    )
    executor.run()


if __name__ == "__main__":
    main()