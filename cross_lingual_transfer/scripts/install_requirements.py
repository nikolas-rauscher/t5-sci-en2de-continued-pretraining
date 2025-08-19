"""
Install requirements for cross-lingual transfer
Installs Wechsel library and dependencies
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_package(package: str):
    """Install a package using pip"""
    try:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")
        raise


def main():
    """Install all required packages for cross-lingual transfer"""
    
    # Required packages
    packages = [
        "wechsel",  # Main cross-lingual transfer library
        "datasets>=2.0.0",  # For German dataset loading
        "transformers>=4.20.0",  # For T5/MT5 models
        "torch>=1.12.0",  # PyTorch
        "pandas",  # Data manipulation
        "numpy",  # Numerical operations
        "tqdm",  # Progress bars
    ]
    
    logger.info("Starting installation of cross-lingual transfer requirements...")
    
    for package in packages:
        install_package(package)
    
    logger.info("All packages installed successfully!")
    
    # Verify Wechsel installation
    try:
        import wechsel
        logger.info(f"Wechsel version: {wechsel.__version__}")
    except ImportError:
        logger.error("Wechsel import failed - installation may be incomplete")
        sys.exit(1)
    
    logger.info("Requirements installation completed successfully!")


if __name__ == "__main__":
    main()