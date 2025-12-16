"""
Configuration for WillSpeak server
"""
from pathlib import Path
import os
import tempfile

# Create data and models directories if they don't exist
data_dir = Path(__file__).parent.parent.parent / "data"
models_dir = Path(__file__).parent.parent.parent / "models"
logs_dir = Path(__file__).parent.parent.parent / "logs"
temp_dir = Path(tempfile.gettempdir()) / "willspeak"

# Ensure directories exist
for directory in [data_dir, models_dir, logs_dir, temp_dir]:
    try:
        if not directory.exists():
            os.makedirs(directory)
    except Exception:
        pass

# User and training directories
user_dir = data_dir / "users"
training_dir = data_dir / "training"
training_data_dir = data_dir / "training_data"

# Ensure these directories exist too
for directory in [user_dir, training_dir, training_data_dir]:
    try:
        if not directory.exists():
            os.makedirs(directory)
    except Exception:
        pass

# Constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB max upload size
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]