# Sber GigaAM-v3 Local Inference

This project provides a professional, local inference pipeline for the **GigaAM-v3** speech-to-text model from SberDevices. It acts as a unified runner for both **NVIDIA CUDA** (Windows/Linux) and **Apple Silicon** (macOS) platforms, optimizing performance for each.

## Features

- **Cross-Platform**: Specialized inference logic for CUDA (using Tensor Cores) and Apple Silicon (using MPS).
- **Batch Processing**: Scans directories for audio files and processes them sequentially with progress tracking.
- **Optimized**: Uses `torch.set_float32_matmul_precision('high')` for NVIDIA Ampere+ GPUs.
- **Output**: Generates a standard JSONL file with filenames and transcriptions.

## Requirements

- Python 3.10 - 3.11
- **Windows/Linux**: NVIDIA GPU with CUDA drivers (CUDA 12.x recommended).
- **macOS**: Apple Silicon (M1/M2/M3) chip.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  Clone the repository.
2.  Install dependencies:

    ```bash
    poetry install
    ```

    *Note: If you have issues with specific torch versions, check `pyproject.toml`.*

## Usage

You can run the interactive main script which will guide you through platform selection and folder input.

```bash
poetry run python src/sber_gigaamv3_local/main.py
```

### Manual Usage

You can also run specific inference scripts if you prefer:

**For CUDA (Windows/Linux):**
```bash
poetry run python src/sber_gigaamv3_local/inference_cuda.py /path/to/audio/dir output.jsonl
```

**For Apple Silicon (macOS):**
```bash
poetry run python src/sber_gigaamv3_local/inference_mps.py /path/to/audio/dir output.jsonl
```

## Project Structure

- `src/sber_gigaamv3_local/main.py`: Main entry point.
- `src/sber_gigaamv3_local/inference_cuda.py`: CUDA-specific logic.
- `src/sber_gigaamv3_local/inference_mps.py`: MPS-specific logic.
- `src/sber_gigaamv3_local/utils.py`: Shared utilities (file finding, saving).
