#!/usr/bin/env python3
"""
Utility script to fix tokenizer format compatibility issues.
Converts any T5-family tokenizer to standard T5 format.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def fix_tokenizer_format(tokenizer_path: str, backup: bool = True, verify: bool = True):
    """
    Fix tokenizer format to ensure T5 compatibility.
    
    Args:
        tokenizer_path: Path to tokenizer directory
        backup: Whether to create backup
        verify: Whether to verify the fix works
    """
    from transformers import T5TokenizerFast, AutoTokenizer
    
    tokenizer_dir = Path(tokenizer_path)
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found at: {tokenizer_json}")
    
    print(f"Fixing tokenizer format at: {tokenizer_dir}")
    
    # Load current config
    with open(tokenizer_json, 'r') as f:
        config = json.load(f)
    
    print("Current pre_tokenizer:")
    print(json.dumps(config.get('pre_tokenizer', {}), indent=2))
    
    # Get T5-standard format
    print("Loading T5-base reference format...")
    ref_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    ref_config = json.loads(ref_tokenizer.backend_tokenizer.to_str())
    
    # Backup if requested
    if backup:
        backup_path = str(tokenizer_json) + '.backup'
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Backup created: {backup_path}")
    
    # Apply fixes
    print("Applying T5-standard format...")
    
    # Fix pre_tokenizer
    old_pre = config.get('pre_tokenizer', {})
    config['pre_tokenizer'] = ref_config['pre_tokenizer']
    print(f"Fixed pre_tokenizer: {old_pre.get('type')} -> {config['pre_tokenizer']['type']}")
    
    # Fix decoder
    old_decoder = config.get('decoder', {})
    config['decoder'] = ref_config['decoder']
    print(f"Fixed decoder: {old_decoder.get('type')} -> {config['decoder']['type']}")
    
    # Save fixed version
    with open(tokenizer_json, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Fixed tokenizer saved")
    
    # Verify if requested
    if verify:
        print("Verifying fix...")
        try:
            test_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            print(f"Verification successful! Vocab size: {len(test_tokenizer)}")
            
            # Test tokenization
            test_text = "Test tokenization with <extra_id_0> tokens."
            tokens = test_tokenizer.encode(test_text)
            decoded = test_tokenizer.decode(tokens)
            print(f"Test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
            
        except Exception as e:
            print(f"Verification failed: {e}")
            raise
    
    print("Tokenizer format fix completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Fix tokenizer format for T5 compatibility")
    parser.add_argument("tokenizer_path", help="Path to tokenizer directory")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    
    args = parser.parse_args()
    
    try:
        fix_tokenizer_format(
            args.tokenizer_path,
            backup=not args.no_backup,
            verify=not args.no_verify
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()