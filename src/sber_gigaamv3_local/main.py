#!/usr/bin/env python3
import os
import sys
import logging
import warnings
from pathlib import Path

# Fix for Apple Silicon (M1/M2/M3) environment
# Must run before any heavy imports or user interaction to avoid duplicate prompts/crashes
sys.path.append(str(Path(__file__).parent.parent))
from sber_gigaamv3_local.platform_fix import apply_mps_fix
apply_mps_fix()

# Suppress noisy warnings from torchaudio/ffmpeg
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

from sber_gigaamv3_local.utils import get_output_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GigaAM-Runner")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set HF_HOME if defined in .env, otherwise let it be default.
# This ensures that if the user set HF_HOME in .env, it's respected by gigaam/transformers.
# Note: load_dotenv() sets them in os.environ, so libraries will pick it up automatically.
hf_home = os.environ.get("HF_HOME")
if hf_home:
    # Make path absolute if it's relative
    hf_home = os.path.abspath(hf_home)
    os.environ["HF_HOME"] = hf_home
    logger.info(f"Using custom Hugging Face cache directory: {hf_home}")
else:
    logger.info("Using default Hugging Face cache directory.")

def print_header():
    print("="*60)
    print("      GigaAM-v3 Local Inference Runner")
    print("="*60)
    print("Supported Platforms: Windows (CUDA), macOS (Apple Silicon)")
    print("")

def get_platform_choice():
    print("Please select your platform:")
    print("1. NVIDIA CUDA (Windows/Linux)")
    print("2. Apple Silicon (macOS M1/M2/M3)")
    
    choice = input("\nEnter choice [1/2]: ").strip()
    return choice

def get_input_directory():
    while True:
        path_str = input("\nEnter the path to the folder containing audio files:\n> ").strip()
        # Remove quotes if user dragged and dropped folder
        path_str = path_str.replace('"', '').replace("'", "")
        path = Path(path_str)
        if path.exists() and path.is_dir():
            return str(path)
        print(f"Error: Directory '{path_str}' does not exist. Please try again.")

def main():
    print_header()
    
    # 1. Platform Selection
    choice = get_platform_choice()
    
    runner = None
    if choice == '1':
        try:
            from sber_gigaamv3_local.inference_cuda import run_cuda_inference
            runner = run_cuda_inference
            print("\nSelected: NVIDIA CUDA Platform")
        except ImportError as e:
            logger.error(f"Failed to import CUDA module: {e}")
            return
    elif choice == '2':
        try:
            from sber_gigaamv3_local.inference_mps import run_mps_inference
            runner = run_mps_inference
            print("\nSelected: Apple Silicon (MPS) Platform")
        except ImportError as e:
            logger.error(f"Failed to import MPS module: {e}")
            return
    else:
        print("Invalid choice. Exiting.")
        return

    # 2. Input Directory
    input_dir = get_input_directory()
    output_file = get_output_filename(input_dir)
    # Save in current working directory or input directory? 
    # Usually safer to save in CWD or let user know. 
    # Let's save in the input directory to be clean? 
    # Or CWD to avoid messing up their source. 
    # Let's save to CWD for visibility.
    output_path = os.path.join(os.getcwd(), output_file)
    
    print(f"\nProcessing files in: {input_dir}")
    print(f"Output will be saved to: {output_path}")
    
    if input("Start inference? (y/n): ").lower() != 'y':
        print("Aborted.")
        return

    # 3. Run Inference
    try:
        runner(input_dir, output_path)
        print("\nDone! ✨")
        print(f"Results saved to: {output_path}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logger.exception("An error occurred during execution")

if __name__ == "__main__":
    main()
