"""
CORRECTED Cross-lingual transfer using Wechsel library
Transfer English T5 model to German by replacing ONLY embeddings
Keeps all trained weights from English model!
"""

import sys
import os
# Add project root to path to resolve src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set cache directory for Wechsel to project directory (MUST BE FIRST!)
cache_dir = '/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/cache/wechsel'
os.environ['WECHSEL_CACHE_DIR'] = cache_dir
os.environ['WECHSEL_CACHE'] = cache_dir  # Alternative env var
os.makedirs(cache_dir, exist_ok=True)

import torch
import logging
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from wechsel import WECHSEL, load_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossLingualTransferFixed:
    def __init__(self, 
                 english_checkpoint_path: str,
                 output_dir: str = "./models/german_transferred_fixed"):
        """
        Initialize cross-lingual transfer (CORRECTED VERSION)
        
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
        self.german_model = None
    
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
        """Setup German tokenizer - using German T5 model"""
        logger.info("Setting up German tokenizer from GermanT5/t5-efficient-gc4-german-base-nl36")
        # Use the German T5 tokenizer
        self.german_tokenizer = AutoTokenizer.from_pretrained('GermanT5/t5-efficient-gc4-german-base-nl36')
        logger.info(f"German tokenizer loaded with vocab size: {len(self.german_tokenizer)}")
    
    def apply_wechsel_transfer(self):
        """Apply Wechsel embedding transfer from English to German"""
        logger.info("Starting Wechsel embedding transfer...")
        
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
        
        # T5 tokenizers don't have .vocab attribute, so we need to create it
        if not hasattr(self.english_tokenizer, 'vocab'):
            # Create vocab mapping for T5 tokenizer
            self.english_tokenizer.vocab = {token: i for i, token in enumerate(self.english_tokenizer.get_vocab().keys())}
        
        if not hasattr(self.german_tokenizer, 'vocab'):
            # Create vocab mapping for German T5 tokenizer  
            self.german_tokenizer.vocab = {token: i for i, token in enumerate(self.german_tokenizer.get_vocab().keys())}
        
        german_embeddings, transfer_info = wechsel.apply(
            self.english_tokenizer,
            self.german_tokenizer, 
            english_embeddings
        )
        logger.info(f"German embeddings shape: {german_embeddings.shape}")
        logger.info(f"Transfer info: {transfer_info}")
        
        # CORRECTED: Keep trained English model, only replace embeddings
        logger.info("CORRECTED APPROACH: Keeping trained English model and replacing only embeddings...")
        self.german_model = self.english_model  # Keep all trained weights!
        
        # Resize embedding layers to match German tokenizer vocab size
        german_vocab_size = len(self.german_tokenizer)
        current_vocab_size = self.german_model.config.vocab_size
        
        if german_vocab_size != current_vocab_size:
            logger.info(f"Resizing embeddings from {current_vocab_size} to {german_vocab_size}")
            self.german_model.resize_token_embeddings(german_vocab_size)
        
        # Update model embeddings with transferred German embeddings
        logger.info("Updating model with transferred German embeddings...")
        self.german_model.get_input_embeddings().weight.data = torch.from_numpy(german_embeddings)
        
        # Also update output embeddings (lm_head) with transferred embeddings
        if hasattr(self.german_model, 'lm_head'):
            self.german_model.lm_head.weight.data = torch.from_numpy(german_embeddings)
        
        logger.info(f"Transfer completed! Kept trained English weights, replaced embeddings for German tokenizer (vocab size: {german_vocab_size})")
        
        return german_embeddings, transfer_info
    
    def save_german_model(self):
        """Save the German-transferred model and tokenizer"""
        logger.info(f"Saving German model to: {self.output_dir}")
        
        # Save model and tokenizer
        self.german_model.save_pretrained(self.output_dir / "model")
        self.german_tokenizer.save_pretrained(self.output_dir / "tokenizer")
        
        # Save transfer metadata
        metadata = {
            "source_checkpoint": str(self.english_checkpoint_path),
            "target_language": "german",
            "transfer_method": "wechsel_corrected",
            "base_model": "t5-base",
            "target_tokenizer": "GermanT5/t5-efficient-gc4-german-base-nl36",
            "methodology": "keep_english_weights_replace_embeddings_only"
        }
        
        torch.save(metadata, self.output_dir / "transfer_metadata.pt")
        logger.info("German model saved successfully")
    
    def run_transfer(self):
        """Complete transfer pipeline"""
        logger.info("Starting CORRECTED cross-lingual transfer pipeline...")
        
        # Load English model
        self.load_english_model()
        
        # Setup German tokenizer  
        self.setup_german_tokenizer()
        
        # Apply Wechsel transfer
        transfer_info = self.apply_wechsel_transfer()
        
        # Save German model
        self.save_german_model()
        
        logger.info("CORRECTED cross-lingual transfer completed successfully!")
        return transfer_info


def main():
    """Main transfer execution"""
    
    # Path to best English checkpoint
    english_checkpoint = "/netscratch/nrauscher/projects/BA-hydra/pretraining_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs/2025-08-13_23-20-56/checkpoints/steps/step-step=640000.ckpt"
    
    # Output directory for German model
    output_dir = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/models/german_transferred_fixed"
    
    # Execute transfer
    transfer = CrossLingualTransferFixed(english_checkpoint, output_dir)
    transfer_info = transfer.run_transfer()
    
    print(f"CORRECTED Transfer completed! Info: {transfer_info}")


if __name__ == "__main__":
    main()