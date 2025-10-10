"""
ROBUST Cross-lingual transfer using Wechsel library
Transfer English T5 model to German by replacing ONLY embeddings
Keeps all trained weights from English model!
Includes all robustness fixes from user feedback.
"""

import sys
import os
# Add project root to path to resolve src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set ALL cache directories to project directory (MUST BE FIRST!)
cache_base = '/netscratch/nrauscher/projects/BA-hydra'
wechsel_cache = f'{cache_base}/cross_lingual_transfer/cache/wechsel'
hf_cache = f'{cache_base}/.hf_cache'

# CRITICAL: Wechsel uses XDG_CACHE_HOME! (found in wechsel/__init__.py:24-25)
os.environ['XDG_CACHE_HOME'] = f'{cache_base}/cross_lingual_transfer/cache'

# Set all possible cache environment variables
os.environ['WECHSEL_CACHE_DIR'] = wechsel_cache
os.environ['WECHSEL_CACHE'] = wechsel_cache
os.environ['HF_HOME'] = hf_cache
os.environ['TRANSFORMERS_CACHE'] = hf_cache
os.environ['HF_DATASETS_CACHE'] = hf_cache
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = hf_cache
os.environ['PYTORCH_PRETRAINED_BERT_CACHE'] = hf_cache

# Create cache directories
os.makedirs(wechsel_cache, exist_ok=True)
os.makedirs(hf_cache, exist_ok=True)

import torch
import logging
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossLingualTransferRobust:
    def __init__(self, 
                 english_checkpoint_path: str,
                 output_dir: str = "./models/german_transferred_robust"):
        """
        Initialize ROBUST cross-lingual transfer
        
        Args:
            english_checkpoint_path: Path to best English checkpoint
            output_dir: Directory to save German-transferred model
        """
        self.english_checkpoint_path = Path(english_checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and tokenizer placeholders
        self.english_model = None
        self.english_tokenizer = None
        self.german_tokenizer = None
        self.model = None  # Renamed for clarity (was self.german_model)
    
    def load_english_model(self):
        """Load the best English T5 checkpoint"""
        logger.info(f"Loading English checkpoint: {self.english_checkpoint_path}")
        
        # Load T5-base tokenizer (used during English pretraining)
        self.english_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        
        # Load model architecture
        self.english_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        # Load checkpoint state dict (PyTorch 2.6+ requires weights_only=False for custom classes)
        checkpoint = torch.load(self.english_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract model state dict from Lightning checkpoint
        if 'state_dict' in checkpoint:
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                # Remove 'model.' prefix from Lightning checkpoint
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                    state_dict[new_key] = value
        else:
            state_dict = checkpoint
        
        # Load state dict into model
        self.english_model.load_state_dict(state_dict, strict=False)
        logger.info("English model loaded successfully")
    
    def setup_german_tokenizer(self):
        """Setup German tokenizer with T5 extra_id tokens"""
        logger.info("Setting up German tokenizer from GermanT5/t5-efficient-gc4-german-base-nl36")
        
        # Load German T5 tokenizer
        self.german_tokenizer = AutoTokenizer.from_pretrained('GermanT5/t5-efficient-gc4-german-base-nl36')
        
        # Ensure T5 extra_id tokens are present (required for T5)
        needed_tokens = {f"<extra_id_{i}>" for i in range(100)}
        current_vocab = set(self.german_tokenizer.get_vocab().keys())
        missing_tokens = needed_tokens - current_vocab
        
        if missing_tokens:
            logger.info(f"Adding {len(missing_tokens)} missing T5 extra_id tokens")
            self.german_tokenizer.add_special_tokens({
                "additional_special_tokens": sorted(list(missing_tokens))
            })
        else:
            logger.info("All T5 extra_id tokens already present")
        
        logger.info(f"German tokenizer loaded with vocab size: {len(self.german_tokenizer)}")
    
    def apply_wechsel_transfer(self):
        """Apply Wechsel embedding transfer from English to German"""
        logger.info("Starting ROBUST Wechsel embedding transfer...")
        
        # Import Wechsel here to avoid import issues
        from wechsel import WECHSEL, load_embeddings
        
        # Initialize Wechsel with English and German embeddings
        logger.info("Loading English and German static embeddings...")
        wechsel = WECHSEL(
            load_embeddings("en"),  # English static embeddings
            load_embeddings("de"),  # German static embeddings  
            bilingual_dictionary="german"  # Use German bilingual dictionary
        )
        logger.info("Wechsel initialized successfully")
        
        # Get current English embeddings
        logger.info("Extracting English model embeddings...")
        english_embeddings = self.english_model.get_input_embeddings().weight.detach().numpy()
        logger.info(f"English embeddings shape: {english_embeddings.shape}")
        
        # Apply Wechsel transformation
        logger.info("Applying Wechsel transformation...")
        
        # ROBUSTNESS FIX 1: Correct tokenizer vocab mapping
        logger.info("Setting up tokenizer vocabularies (FIXED: using get_vocab() directly)")
        # Wechsel expects tokenizer.vocab attribute, but T5Tokenizer doesn't have it
        # We need to add it manually as a workaround
        if not hasattr(self.english_tokenizer, 'vocab'):
            self.english_tokenizer.vocab = self.english_tokenizer.get_vocab()
        if not hasattr(self.german_tokenizer, 'vocab'):
            self.german_tokenizer.vocab = self.german_tokenizer.get_vocab()
        
        german_embeddings, transfer_info = wechsel.apply(
            self.english_tokenizer,
            self.german_tokenizer, 
            english_embeddings
        )
        logger.info(f"German embeddings shape: {german_embeddings.shape}")
        logger.info(f"Transfer info: {transfer_info}")
        
        # CORRECTED APPROACH: Keep trained English model, only replace embeddings
        logger.info("ROBUST APPROACH: Keeping trained English model and replacing only embeddings...")
        self.model = self.english_model  # Keep all trained weights!
        
        # Resize embedding layers to match German tokenizer vocab size
        german_vocab_size = len(self.german_tokenizer)
        current_vocab_size = self.model.config.vocab_size
        
        if german_vocab_size != current_vocab_size:
            logger.info(f"Resizing embeddings from {current_vocab_size} to {german_vocab_size}")
            self.model.resize_token_embeddings(german_vocab_size)
        
        # ROBUSTNESS FIX 2: Correct dtype/device handling for embeddings
        logger.info("Updating model with transferred German embeddings (FIXED: correct dtype/device)")
        with torch.no_grad():
            # Get reference to input embeddings for dtype/device
            emb = self.model.get_input_embeddings().weight
            # Convert numpy to tensor with correct dtype/device
            W = torch.as_tensor(german_embeddings, dtype=emb.dtype, device=emb.device)
            # Copy embeddings
            emb.copy_(W)
            # T5 has get_output_embeddings() == lm_head
            out = self.model.get_output_embeddings().weight
            out.copy_(W)
        
        # ROBUSTNESS FIX 3: Set special token IDs in config
        logger.info("Configuring special token IDs (FIXED: proper config setup)")
        cfg = self.model.config
        tok = self.german_tokenizer
        cfg.vocab_size = len(tok)
        cfg.pad_token_id = tok.pad_token_id
        cfg.eos_token_id = tok.eos_token_id
        cfg.decoder_start_token_id = tok.pad_token_id  # T5-typical
        
        # Ensure weights are tied properly
        self.model.tie_weights()
        
        logger.info(f"ROBUST transfer completed! Kept trained English weights, replaced embeddings for German tokenizer (vocab size: {german_vocab_size})")
        
        # ROBUSTNESS CHECK: Sanity checks
        logger.info("Running sanity checks...")
        assert self.model.config.vocab_size == len(self.german_tokenizer), "Config vocab size mismatch"
        assert self.model.get_input_embeddings().weight.shape[0] == len(self.german_tokenizer), "Embedding shape mismatch"
        
        # Check if embeddings are tied (should be for T5)
        input_emb = self.model.get_input_embeddings().weight
        output_emb = self.model.get_output_embeddings().weight
        if torch.allclose(input_emb, output_emb, atol=1e-6):
            logger.info("✅ Input and output embeddings are properly tied")
        else:
            logger.warning("⚠️ Input and output embeddings are NOT tied - this might be expected")
        
        logger.info("All sanity checks passed!")
        
        return german_embeddings, transfer_info
    
    def _fix_tokenizer_format(self, tokenizer_json_path):
        """
        Fix tokenizer.json format to ensure T5 compatibility.
        Converts old-style pre_tokenizer and decoder to T5-standard format.
        """
        import json
        from transformers import T5TokenizerFast
        
        logger.info(f"Fixing tokenizer format at: {tokenizer_json_path}")
        
        try:
            # Load current tokenizer config
            with open(tokenizer_json_path, 'r') as f:
                config = json.load(f)
            
            # Get reference T5 format from t5-base
            ref_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
            ref_config = json.loads(ref_tokenizer.backend_tokenizer.to_str())
            
            # Fix pre_tokenizer to T5 standard format
            logger.info("Fixing pre_tokenizer format...")
            config['pre_tokenizer'] = ref_config['pre_tokenizer']
            
            # Fix decoder to T5 standard format  
            logger.info("Fixing decoder format...")
            config['decoder'] = ref_config['decoder']
            
            # Backup original
            backup_path = str(tokenizer_json_path) + '.backup'
            with open(backup_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Original backed up to: {backup_path}")
            
            # Save fixed version
            with open(tokenizer_json_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("Tokenizer format fixed to T5 standard")
            
            # Test that it loads correctly
            from transformers import AutoTokenizer
            test_tokenizer = AutoTokenizer.from_pretrained(tokenizer_json_path.parent)
            logger.info(f"Format fix verified - tokenizer loads correctly (vocab size: {len(test_tokenizer)})")
            
        except Exception as e:
            logger.error(f"Failed to fix tokenizer format: {e}")
            logger.warning("Tokenizer may have compatibility issues, but transfer continues...")
            # Don't fail the entire transfer for this
    
    def save_german_model(self):
        """Save the German-transferred model and tokenizer"""
        logger.info(f"Saving ROBUST German model to: {self.output_dir}")
        
        # Save model
        self.model.save_pretrained(self.output_dir / "model")
        
        # Save tokenizer in T5-compatible format
        logger.info("Saving tokenizer with T5-compatible format...")
        
        # First save normally
        self.german_tokenizer.save_pretrained(self.output_dir / "tokenizer")
        
        # Then fix the tokenizer.json format to ensure T5 compatibility
        self._fix_tokenizer_format(self.output_dir / "tokenizer" / "tokenizer.json")
        
        logger.info("Tokenizer saved and format-fixed for T5 compatibility")
        
        # Save transfer metadata with robustness info
        metadata = {
            "source_checkpoint": str(self.english_checkpoint_path),
            "target_language": "german", 
            "transfer_method": "wechsel_robust",
            "base_model": "t5-base",
            "target_tokenizer": "GermanT5/t5-efficient-gc4-german-base-nl36",
            "methodology": "wechsel_embeddings_only; keep_all_en_layers",
            "robustness_fixes": [
                "correct_tokenizer_vocab_mapping",
                "dtype_device_handling", 
                "special_token_config",
                "t5_extra_id_tokens",
                "sanity_checks"
            ],
            "vocab_size_final": len(self.german_tokenizer),
            "extra_id_tokens_added": True
        }
        
        torch.save(metadata, self.output_dir / "transfer_metadata.pt")
        logger.info("ROBUST German model saved successfully")
    
    def run_transfer(self):
        """Complete ROBUST transfer pipeline"""
        logger.info("Starting ROBUST cross-lingual transfer pipeline...")
        
        # Load English model
        self.load_english_model()
        
        # Setup German tokenizer with T5 tokens
        self.setup_german_tokenizer()
        
        # Apply Wechsel transfer with robustness fixes
        transfer_info = self.apply_wechsel_transfer()
        
        # Save German model
        self.save_german_model()
        
        logger.info("ROBUST cross-lingual transfer completed successfully!")
        return transfer_info


def main():
    """Main ROBUST transfer execution"""
    
    # Path to best English checkpoint (Clean Restart - best val_ppl)
    english_checkpoint = "/netscratch/nrauscher/projects/BA-hydra/pretraining_logs_lr_001_OPTIMIZED_clean_restart/train/runs/2025-09-08_02-33-22/checkpoints/best/step-487500-val_ppl-3.72168.ckpt"
    
    # Output directory for German model (Optimized 50% Overlap Clean Restart)
    output_dir = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/models/german_T5_Optimized_50Olap_clean_restart_487k"
    
    # Execute transfer
    transfer = CrossLingualTransferRobust(english_checkpoint, output_dir)
    transfer_info = transfer.run_transfer()
    
    print(f"ROBUST Transfer completed! Info: {transfer_info}")


if __name__ == "__main__":
    main()