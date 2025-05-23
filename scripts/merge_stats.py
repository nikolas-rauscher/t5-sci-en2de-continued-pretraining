import hydra
from omegaconf import DictConfig
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.stats import StatsMerger


@hydra.main(config_path="../configs", config_name="cleaning/datatrove")
def main(cfg: DictConfig) -> None:

    stats_input_folder = cfg.cleaning.paths.dst + "/stats"
    stats_output_folder = cfg.cleaning.paths.dst + "/stats_merged" 

    pipeline = [
        StatsMerger(input_folder=stats_input_folder, output_folder=stats_output_folder, remove_input=False)
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1, 
        workers=1, 
    )
    executor.run()


if __name__ == "__main__":
    main() 