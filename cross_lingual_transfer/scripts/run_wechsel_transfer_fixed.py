#!/usr/bin/env python
"""
Wrapper script to run CORRECTED Wechsel transfer with proper imports
"""

import sys
import os

# Add project root to path for src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import src modules to register custom classes
try:
    from src.models.t5_module import T5LitModule
    from src.data.t5_datamodule import T5DataModule
    from src.data.t5_datamodule_token_windows import T5DataModuleTokenWindows
    from src.data.t5_datamodule_text_windows import T5DataModuleTextWindows
    from src.data.t5_dataset_materialized_window import T5MaterializedWindowDataset
    print("Successfully imported src modules")
except ImportError as e:
    print(f"Warning: Could not import src modules: {e}")
    print("Continuing anyway...")

# Now run the CORRECTED transfer script
from cross_lingual_transfer.scripts.wechsel_transfer_fixed import main

if __name__ == "__main__":
    main()