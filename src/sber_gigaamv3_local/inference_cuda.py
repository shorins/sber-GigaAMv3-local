import torch
import gigaam
import logging
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import time
import torchaudio

logger = logging.getLogger(__name__)

class CudaInference:
    def __init__(self, model_name: str = "v3_e2e_rnnt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            logger.warning("CUDA is not available! Fallback to CPU, but this module is intended for CUDA.")
        
        # Optimization for Tensor Cores (Ampere/Blackwell/etc)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            prop = torch.cuda.get_device_properties(self.device)
            logger.info(f"Using CUDA device: {prop.name}")
        
        logger.info(f"Loading model {model_name}...")
        self.model = gigaam.load_model(model_name)
        
        # Move model to device explicitly if possible
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
             self.model.model = self.model.model.to(self.device)
        
        logger.info("Model loaded and moved to CUDA.")

    def transcribe_files(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        results = []
        
        # Warm-up (optional but good practice)
        if file_paths:
             logger.info("Warming up CUDA...")
             try:
                 _ = self.model.transcribe(str(file_paths[0]))
                 torch.cuda.synchronize()
             except Exception as e:
                 logger.warning(f"Warmup failed: {e}")

        logger.info(f"Starting inference on {len(file_paths)} files...")
        for file_path in tqdm(file_paths, desc="Transcribing (CUDA)"):
            try:
                start_time = time.time()
                
                # Check duration to decide method
                use_longform = False
                try:
                    metadata = torchaudio.info(str(file_path))
                    duration = metadata.num_frames / metadata.sample_rate
                    if duration > 20.0:
                        use_longform = True
                except:
                    pass

                if use_longform:
                    transcription_result = self.model.transcribe_longform(str(file_path))
                    if isinstance(transcription_result, list):
                        text = " ".join([seg.get('text', '') for seg in transcription_result])
                    else:
                        text = str(transcription_result)
                else:
                    try:
                        text = self.model.transcribe(str(file_path))
                    except Exception as e:
                        if "Too long wav file" in str(e):
                            logger.info(f"File {file_path.name} is too long, switching to longform...")
                            transcription_result = self.model.transcribe_longform(str(file_path))
                            if isinstance(transcription_result, list):
                                text = " ".join([seg.get('text', '') for seg in transcription_result])
                            else:
                                text = str(transcription_result)
                        else:
                            raise e

                torch.cuda.synchronize() # Wait for kernels to finish for accurate timing/flow
                duration = time.time() - start_time
                
                results.append({
                    "file": file_path.name,
                    "path": str(file_path),
                    "text": text,
                    "inference_time": duration
                })
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    "file": file_path.name,
                    "path": str(file_path),
                    "error": str(e)
                })
                
        return results

def run_cuda_inference(input_dir: str, output_file: str):
    from .utils import find_audio_files, save_jsonl
    
    files = find_audio_files(input_dir)
    if not files:
        logger.error("No audio files found.")
        return

    inferencer = CudaInference()
    results = inferencer.transcribe_files(files)
    save_jsonl(results, output_file)

if __name__ == "__main__":
    # Example usage for testing
    import sys
    if len(sys.argv) > 2:
        run_cuda_inference(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python inference_cuda.py <input_dir> <output_file>")
