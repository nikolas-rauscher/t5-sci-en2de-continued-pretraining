import os
import sys

# 1) Projekt-Root und lokale Datatrove-Quelle ins PYTHONPATH
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, proj_root)
# sys.path.insert(0, os.path.join(proj_root, "external", "datatrove", "src"))

import hydra
from omegaconf import DictConfig, OmegaConf # OmegaConf not strictly needed here for this fix
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
# Import PerplexityStats, not CCNetPerplexityStats, based on your config
from datatrove.pipeline.stats import (
    # WordsContaminationStats, # Not used in your pipeline
    DocStats,
    LangStats,
    LineStats,
    ParagraphStats,
    SentenceStats,
    TokenStats,
    WordStats,
    CCNetPerplexityStats, # Changed from CCNetPerplexityStats to match config
)

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="stats/compute_stats_datatrove"
)

def main(cfg: DictConfig) -> None:
    base_dst_from_config = cfg.stats.paths.dst
    modules_config = cfg.stats.pipeline.stats_modules # Enthält jetzt die korrekten groups_to_compute

    doc_out  = modules_config.doc_stats.output_folder
    line_out = modules_config.line_stats.output_folder
    para_out = modules_config.paragraph_stats.output_folder
    sent_out = modules_config.sentence_stats.output_folder
    tok_out  = modules_config.token_stats.output_folder
    word_out = modules_config.word_stats.output_folder
    lang_out = modules_config.lang_stats.output_folder
    perp_out = modules_config.perplexity_stats.output_folder

    pipeline = [
        ParquetReader(
            data_folder   = cfg.stats.paths.src_dir,
            glob_pattern  = cfg.stats.paths.src_pattern,
            limit         = cfg.stats.limit_documents,
        ),
        DocStats(
            output_folder   = doc_out,
            groups_to_compute = modules_config.doc_stats.groups_to_compute,
        ),
        LineStats(
            output_folder = line_out,
            groups_to_compute = modules_config.line_stats.groups_to_compute,
        ),
        ParagraphStats(
            output_folder = para_out,
            groups_to_compute = modules_config.paragraph_stats.groups_to_compute,
        ),
        SentenceStats(
            output_folder = sent_out,
            language = modules_config.sentence_stats.language,
            groups_to_compute = modules_config.sentence_stats.groups_to_compute,
        ),
        TokenStats(
            output_folder = tok_out,
            tokenizer_name_or_path = modules_config.token_stats.tokenizer_name_or_path,
            groups_to_compute = modules_config.token_stats.groups_to_compute,
        ),
        WordStats(
            output_folder = word_out,
            language = modules_config.word_stats.language,
            groups_to_compute = modules_config.word_stats.groups_to_compute,
        ),
        LangStats(
            output_folder = lang_out,
            language = modules_config.lang_stats.language,
            groups_to_compute = modules_config.lang_stats.groups_to_compute,
        ),
        CCNetPerplexityStats(
            output_folder = perp_out,
            model_dataset = modules_config.perplexity_stats.model_dataset,
            language = modules_config.perplexity_stats.language,
            groups_to_compute = modules_config.perplexity_stats.groups_to_compute,
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline = pipeline,
        tasks    = cfg.stats.tasks,
        workers  = cfg.stats.workers,
    )
    executor.run()

if __name__ == "__main__":
    main()