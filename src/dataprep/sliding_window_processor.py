
import logging
from typing import Dict, List, Tuple
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.logging import logger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SlidingWindowProcessor(PipelineStep):
    """
    computes token counts with t5 sentencepiece
    stores token count in document.metadata
    windows computed on-the-fly during training
    """
    
    type = "token_counts"
    name = "token-count-processor"
    
    def __init__(
        self,
        tokenizer_name_or_path: str = "t5-small",
        max_length: int = 512,
        overlap_size: int = 256,
        log_to_wandb: bool = False,
        wandb_project: str = "BA-DataTrove",
        wandb_group: str = "token-count-preprocessing"
    ):
        super().__init__()
        
        self.max_length = max_length
        self.overlap_size = overlap_size
        self.stride = max_length - overlap_size
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        
        # init tokenizer
        try:
            from transformers import T5TokenizerFast
            import transformers
            transformers.logging.set_verbosity_error()
            
            self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name_or_path)
            logger.info(f"loaded t5 sentencepiece tokenizer: {tokenizer_name_or_path}")
        except Exception as e:
            logger.error(f"failed to load tokenizer {tokenizer_name_or_path}: {e}")
            raise
        
        # metrics tracking
        self.processed_docs = 0
        self.total_tokens = 0
        self.short_docs = 0
        self.long_docs = 0
        self.total_windows = 0
        
        # w&b init
        self.wandb_run = None
        if self.log_to_wandb:
            logger.info("w&b will be initialized per worker rank")
    
    def _compute_token_count(self, text: str) -> int:
        """compute token count with t5 sentencepiece tokenizer"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def _compute_window_count(self, token_count: int) -> int:
        """compute number of windows based on token count"""
        if token_count <= self.max_length:
            return 1
        else:
            return max(1, (token_count - self.overlap_size) // self.stride + 1)
    
    def _update_document_metadata(self, doc: Document, token_count: int) -> None:
        """update document metadata with token count and window config"""
        
        if not hasattr(doc, 'metadata') or doc.metadata is None:
            doc.metadata = {}
        
        window_count = self._compute_window_count(token_count)
        
        doc.metadata.update({
            "t5_sentencepiece_token_count": token_count,
            "t5_sentencepiece_tokenizer_path": str(self.tokenizer.name_or_path),
            
            "t5_sliding_window_config": {
                "max_length": self.max_length,
                "overlap_size": self.overlap_size,
                "stride": self.stride,
                "tokenizer_class": str(self.tokenizer.__class__.__name__),
                "tokenizer_path": str(self.tokenizer.name_or_path)
            },
            
            "t5_estimated_window_count": window_count,
            "t5_document_type": "short" if window_count == 1 else "long",
            
            "t5_estimated_total_window_tokens": window_count * self.max_length,
            "t5_data_utilization": min(1.0, token_count / (window_count * self.max_length))
        })
    
    def _log_document_metrics(self, doc: Document, token_count: int) -> None:
        """log metrics for processed document"""
        
        window_count = self._compute_window_count(token_count)
        is_short = window_count == 1
        
        self.processed_docs += 1
        self.total_tokens += token_count
        self.total_windows += window_count
        
        if is_short:
            self.short_docs += 1
        else:
            self.long_docs += 1
        
        # log to w&b periodically
        if self.log_to_wandb and self.processed_docs % 1000 == 0 and hasattr(self, 'wandb_run') and self.wandb_run:
            try:
                avg_windows_per_doc = self.total_windows / self.processed_docs
                avg_tokens_per_doc = self.total_tokens / self.processed_docs
                
                wandb.log({
                    "processed_documents": self.processed_docs,
                    "total_estimated_windows": self.total_windows,
                    "short_documents": self.short_docs,
                    "long_documents": self.long_docs,
                    "avg_windows_per_doc": avg_windows_per_doc,
                    "avg_tokens_per_doc": avg_tokens_per_doc,
                    "short_doc_ratio": self.short_docs / self.processed_docs,
                    "data_utilization": self.total_tokens / (self.total_windows * self.max_length)
                })
            except Exception as e:
                logger.warning(f"w&b logging failed: {e}")
                self.log_to_wandb = False
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        """main processing loop for token count computation"""
        
        # init w&b only in rank 0 process
        if self.log_to_wandb and rank == 0 and self.wandb_run is None:
            try:
                self.wandb_run = wandb.init(
                    project="BA-DataTrove",
                    group="token-count-preprocessing", 
                    job_type=f"token-count-preprocessing-rank-{rank}",
                    tags=["preprocessing", "token-counts", "t5-sentencepiece"],
                    config={
                        "max_length": self.max_length,
                        "overlap_size": self.overlap_size,
                        "stride": self.stride,
                        "rank": rank,
                        "world_size": world_size
                    },
                    reinit=True
                )
                logger.info(f"w&b initialized for rank {rank}: {self.wandb_run.url}")
            except Exception as e:
                logger.warning(f"w&b initialization failed for rank {rank}: {e}")
                self.log_to_wandb = False
        elif rank != 0:
            self.log_to_wandb = False
            
        logger.info(f"starting token count preprocessing (rank {rank}/{world_size})")
        logger.info(f"window config: max_length={self.max_length}, overlap={self.overlap_size}, stride={self.stride}")
        logger.info(f"windows will be computed on-the-fly during training")
        
        for doc in data:
            self.stat_update("total_docs")
            
            with self.track_time():
                try:
                    token_count = self._compute_token_count(doc.text)
                    
                    self._update_document_metadata(doc, token_count)
                    
                    self._log_document_metrics(doc, token_count)
                    
                    window_count = self._compute_window_count(token_count)
                    self.stat_update("windows_estimated", window_count)
                    self.stat_update("tokens_processed", token_count)
                    
                    if window_count == 1:
                        self.stat_update("short_documents")
                    else:
                        self.stat_update("long_documents")
                    
                    yield doc
                    
                except Exception as e:
                    logger.warning(f"error processing document {doc.id}: {e}")
                    self.stat_update("processing_errors")
                    continue
        
        # final summary
        if self.processed_docs > 0:
            avg_windows = self.total_windows / self.processed_docs
            avg_tokens = self.total_tokens / self.processed_docs
            data_utilization = self.total_tokens / (self.total_windows * self.max_length)
            
            logger.info(f"token count preprocessing completed")
            logger.info(f"processed {self.processed_docs} documents")
            logger.info(f"token count preprocessing completed!")
            logger.info(f"processed {self.processed_docs} documents")
            logger.info(f"processed {self.total_tokens} tokens ({avg_tokens:.0f} per doc)")
            logger.info(f"estimated {self.total_windows} windows ({avg_windows:.1f} per doc)")
            logger.info(f"data utilization: {data_utilization:.1%}")
            logger.info(f"short docs: {self.short_docs} ({self.short_docs/self.processed_docs:.1%})")
            logger.info(f"windows will be computed on-the-fly during training")
            
            # final w&b log
            if self.log_to_wandb and hasattr(self, 'wandb_run') and self.wandb_run and rank == 0:
                try:
                    wandb.log({
                        "final_processed_documents": self.processed_docs,
                        "final_total_estimated_windows": self.total_windows,
                        "final_avg_windows_per_doc": avg_windows,
                        "final_avg_tokens_per_doc": avg_tokens,
                        "final_data_utilization": data_utilization,
                        "final_short_doc_ratio": self.short_docs / self.processed_docs
                    })
                    wandb.finish()
                    logger.info(f"📊 W&B session finished for rank {rank}")
                except Exception as e:
                    logger.warning(f"Final W&B logging failed: {e}") 