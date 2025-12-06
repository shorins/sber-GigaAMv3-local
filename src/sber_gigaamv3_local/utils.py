import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.opus'}

def find_audio_files(directory: str) -> List[Path]:
    """Recursively find all audio files in a directory."""
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    audio_files = []
    for p in path.rglob('*'):
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(p)
            
    return sorted(audio_files)

def save_jsonl(data: List[Dict[str, Any]], output_path: str):
    """Save a list of dictionaries to a JSONL file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Successfully saved transcriptions to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSONL: {e}")
        raise

def get_output_filename(input_dir: str) -> str:
    """Generate an output filename based on the input directory name."""
    dir_name = Path(input_dir).name
    return f"{dir_name}_transcriptions.jsonl"
