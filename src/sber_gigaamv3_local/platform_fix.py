import os
import sys

def apply_mps_fix():
    """
    Checks if running on macOS and if DYLD_LIBRARY_PATH is correctly set for 
    local FFmpeg (ffmpeg@6) usage with torchcodec.
    
    If the environment variable is missing the required path, this function 
    will restart the current process with the updated environment.
    """
    if sys.platform != "darwin":
        return

    ffmpeg_path = "/opt/homebrew/opt/ffmpeg@6/lib"
    current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")

    if ffmpeg_path not in current_dyld:
        # Update env
        new_dyld = f"{ffmpeg_path}:{current_dyld}" if current_dyld else ffmpeg_path
        os.environ["DYLD_LIBRARY_PATH"] = new_dyld
        
        # We must restart the process for dlopen to pick up the new DYLD_LIBRARY_PATH
        # This will re-run the script with the exact same arguments
        try:
             print("[MPS Fix] Restarting process with updated DYLD_LIBRARY_PATH...")
             os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
             # If execv fails, proceed (unlikely to work but prevent crash)
             print(f"[MPS Fix] Warning: Failed to restart process: {e}")
             pass
