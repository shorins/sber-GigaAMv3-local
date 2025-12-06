from .platform_fix import apply_mps_fix
# Ensure env is correct (in case run directly)
apply_mps_fix()

import torch
import gigaam
import logging
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import time
import torchaudio

logger = logging.getLogger(__name__)

class MPSInference:
    def __init__(self, model_name: str = "v3_e2e_rnnt"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if self.device.type != "mps":
            logger.warning("MPS (Metal Performance Shaders) is not available! Fallback to CPU, but this module is intended for Apple Silicon.")
        else:
            logger.info("Using Apple Silicon (MPS) acceleration.")

        logger.info(f"Loading model {model_name}...")
        self.model = gigaam.load_model(model_name)
        
        # Move model to device
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
             self.model.model = self.model.model.to(self.device)
             
        logger.info(f"Model loaded and moved to {self.device}.")

    def transcribe_files(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        results = []
        
        # Warm-up
        if file_paths:
             logger.info("Warming up MPS...")
             try:
                 _ = self.model.transcribe(str(file_paths[0]))
                 if torch.backends.mps.is_available():
                     torch.mps.synchronize()
             except Exception as e:
                 logger.warning(f"Warmup failed: {e}")

        logger.info(f"Starting inference on {len(file_paths)} files...")
        for file_path in tqdm(file_paths, desc="Transcribing (MPS)"):
            try:
                start_time = time.time()
                
                # Check duration to decide method (heuristic: > 20s uses longform to be safe)
                # Note: Default limit is often 30s, but let's be conservative.
                use_longform = False
                try:
                    metadata = torchaudio.info(str(file_path))
                    duration = metadata.num_frames / metadata.sample_rate
                    if duration > 20.0:
                        use_longform = True
                except:
                    pass # Fallback to try/except strategy

                if use_longform:
                    text = self.custom_longform_transcribe(str(file_path))
                else:
                    try:
                        text = self.model.transcribe(str(file_path))
                    except Exception as e:
                        if "Too long wav file" in str(e):
                            logger.info(f"File {file_path.name} is too long, switching to custom longform...")
                            text = self.custom_longform_transcribe(str(file_path))
                        else:
                            raise e


                
                # Sync for accurate timing on MPS
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                    
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

    def custom_longform_transcribe(self, file_path: str, chunk_duration: int = 20) -> str:
        """
        Manually splits audio into chunks and transcribes them to bypass internal issues.
        """
        import torchaudio
        import tempfile
        import os
        import time

        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Resample to 16k if needed (GigaAM preferred)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Work with mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            total_samples = waveform.size(1)
            chunk_samples = chunk_duration * sample_rate
            
            transcriptions = []
            
            # Ensure temp dir exists
            with tempfile.TemporaryDirectory() as temp_dir:
                for start in range(0, total_samples, chunk_samples):
                    end = min(start + chunk_samples, total_samples)
                    chunk = waveform[:, start:end]
                    
                    # Save chunk
                    temp_chunk_path = os.path.join(temp_dir, f"chunk_{start}.wav")
                    torchaudio.save(temp_chunk_path, chunk, sample_rate)
                    
                    try:
                        # Transcribe chunk
                        chunk_text = self.model.transcribe(temp_chunk_path)
                        if chunk_text:
                            transcriptions.append(chunk_text)
                    except Exception as e:
                        logger.warning(f"Failed to transcribe chunk {start}: {e}")
                        
            return " ".join(transcriptions)
            
        except Exception as e:
            logger.error(f"Custom longform failed: {e}")
            return ""

def run_mps_inference(input_dir: str, output_file: str):
    from .utils import find_audio_files, save_jsonl
    
    files = find_audio_files(input_dir)
    if not files:
        logger.error("No audio files found.")
        return

    inferencer = MPSInference()
    results = inferencer.transcribe_files(files)
    save_jsonl(results, output_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        run_mps_inference(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python inference_mps.py <input_dir> <output_file>")
