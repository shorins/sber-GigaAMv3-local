# Apple Silicon (M1/M2/M3) Troubleshooting Guide

This document details the specific challenges encountered when running the **GigaAM-v3** model locally on macOS (Apple Silicon) and the solutions implemented to resolve them.

## Summary of Issues
1. **TorchCodec & FFmpeg Compatibility**: `torchcodec` crashed due to version mismatches with the system's default FFmpeg.
2. **Dynamic Library Loading**: Python could not find the installed ffmpeg libraries despite them being present.
3. **Broken Dependencies**: Homebrew's `ffmpeg@6` installation had broken links to `libbluray`.
4. **Long-form Transcription Failure**: The built-in `transcribe_longform` method returned empty text on MPS (Metal Performance Shaders).

---

## Detailed Breakdown

### 1. TorchCodec and FFmpeg Versioning
**Problem:**  
The project relies on `torchcodec` for audio decoding. However, `torchcodec` currently supports FFmpeg versions 4, 5, 6, and 7. macOS Homebrew defaults to installing FFmpeg 8 (or newer), causing the error:
> `name 'AudioDecoder' is not defined`  
> `RuntimeError: Could not load libtorchcodec`

**Solution:**  
We forced the use of **FFmpeg 6**.
```bash
brew install ffmpeg@6
```

### 2. DYLD_LIBRARY_PATH on macOS
**Problem:**  
Even with `ffmpeg@6` installed, Python's dynamic linker (`dlopen`) fails to locate the libraries because they are in a non-standard location (`/opt/homebrew/opt/ffmpeg@6/lib`) and macOS System Integrity Protection (SIP) sanitizes certain environment variables.

**Solution:**  
We implemented **Run-time Self-Restart Logic** in `src/sber_gigaamv3_local/inference_mps.py`. 
The script now:
1. Checks if `DYLD_LIBRARY_PATH` contains the `ffmpeg@6` path on startup.
2. If missing, it updates the environment variable.
3. **Restarts the process** (`os.execv`) so the dynamic linker picks up the new path before loading any libraries.

```python
# Snippet from inference_mps.py
if sys.platform == "darwin":
    ffmpeg_path = "/opt/homebrew/opt/ffmpeg@6/lib"
    # ... checks env ...
    if ffmpeg_path not in current_dyld:
         # ... sets env ...
         os.execv(sys.executable, [sys.executable] + sys.argv)
```

### 3. Broken `libbluray` Dependency
**Problem:**  
During debugging, we found that `ffmpeg@6` was failing to load because it expected `libbluray.2.dylib`, but the system had upgraded to `libbluray.3.dylib`.

**Solution:**  
Reinstall the package to rebuild links against current system compatible versions:
```bash
brew reinstall ffmpeg@6
```

### 4. Empty Results for Long Audio (MPS Issue)
**Problem:**  
While standard `transcribe()` worked perfectly for short audio on MPS (GPU), the `transcribe_longform()` method (used for files > 20s) consistently returned empty strings or failed silently. This appears to be an internal issue with the GigaAM segmentation pipeline interacting with MPS tensors.

**Solution:**  
We bypassed the internal long-form method by implementing `custom_longform_transcribe` in `MPSInference`.
*   **Logic**: It manually splits the audio file into 20-second blocks using `torchaudio`.
*   **Processing**: Each block is saved to a temp file and processed via the standard `transcribe()` method (which is known to work).
*   **Result**: The segments are joined to form the final text.

## How to Run
No special manual configuration is needed anymore. The script handles the environment setup automatically.

```bash
poetry run python src/sber_gigaamv3_local/main.py
```
