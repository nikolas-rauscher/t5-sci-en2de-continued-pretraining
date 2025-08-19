"""
Cross-lingual transfer using Wechsel library
Transfer English T5 model to German by replacing embeddings
"""

import torch
import logging
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
from wechsel import WECHSEL, load_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossLingualTransfer:
    def __init__(self, 
                 english_checkpoint_path: str,
                 output_dir: str = "./models/german_transferred"):
        """
        Initialize cross-lingual transfer
        
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
        
        # Load checkpoint state dict
        checkpoint = torch.load(self.english_checkpoint_path, map_location='cpu')
        
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
        """Setup German tokenizer - using multilingual T5 for German support"""
        logger.info("Setting up German tokenizer")
        # Use multilingual T5 which includes German tokens
        self.german_tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
        logger.info("German tokenizer loaded")
    
    def apply_wechsel_transfer(self):
        """Apply Wechsel embedding transfer from English to German"""
        logger.info("Starting Wechsel embedding transfer...")
        
        # Initialize Wechsel with English and German embeddings
        wechsel = WECHSEL(
            load_embeddings("en"),  # English static embeddings
            load_embeddings("de"),  # German static embeddings  
            bilingual_dictionary="german"  # Use German bilingual dictionary
        )
        
        # Get current English embeddings
        english_embeddings = self.english_model.get_input_embeddings().weight.detach().numpy()
        
        # Apply Wechsel transformation
        logger.info("Applying Wechsel transformation...")
        german_embeddings, transfer_info = wechsel.apply(
            self.english_tokenizer,
            self.german_tokenizer, 
            english_embeddings
        )
        
        # Create new model with German tokenizer architecture
        self.german_model = T5ForConditionalGeneration.from_pretrained('google/mt5-base')
        
        # Copy English model parameters to German model (except embeddings)
        english_state = self.english_model.state_dict()
        german_state = self.german_model.state_dict()
        
        for name, param in english_state.items():
            if name in german_state and 'embed' not in name.lower():
                # Copy non-embedding parameters
                if param.shape == german_state[name].shape:
                    german_state[name] = param
                    
        # Set transferred German embeddings
        self.german_model.get_input_embeddings().weight.data = torch.from_numpy(german_embeddings)
        
        # Also update output embeddings if they exist and are tied
        if hasattr(self.german_model, 'lm_head') and self.german_model.config.tie_word_embeddings:
            self.german_model.lm_head.weight.data = torch.from_numpy(german_embeddings)
        
        logger.info(f"Wechsel transfer completed. Transfer info: {transfer_info}")
        return transfer_info
    
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
            "transfer_method": "wechsel",
            "base_model": "t5-base",
            "target_tokenizer": "mt5-base"
        }
        
        torch.save(metadata, self.output_dir / "transfer_metadata.pt")
        logger.info("German model saved successfully")
    
    def run_transfer(self):
        """Complete transfer pipeline"""
        logger.info("Starting cross-lingual transfer pipeline...")
        
        # Load English model
        self.load_english_model()
        
        # Setup German tokenizer  
        self.setup_german_tokenizer()
        
        # Apply Wechsel transfer
        transfer_info = self.apply_wechsel_transfer()
        
        # Save German model
        self.save_german_model()
        
        logger.info("Cross-lingual transfer completed successfully!")
        return transfer_info


def main():
    """Main transfer execution"""
    
    # Path to best English checkpoint
    english_checkpoint = "/netscratch/nrauscher/projects/BA-hydra/pretraining_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs/2025-08-13_23-20-56/checkpoints/steps/step-step=640000.ckpt"
    
    # Output directory for German model
    output_dir = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/models/german_transferred"
    
    # Execute transfer
    transfer = CrossLingualTransfer(english_checkpoint, output_dir)
    transfer_info = transfer.run_transfer()
    
    print(f"Transfer completed! Info: {transfer_info}")


if __name__ == "__main__":
    main()