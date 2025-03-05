"""
WillSpeak Server - FastAPI Application

A speech enhancement server that processes audio files to improve clarity,
particularly designed to help individuals with speech impediments.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, BackgroundTasks, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
import numpy as np
import io
import soundfile as sf
import os
import sys
# Import Path explicitly from pathlib to avoid conflicts with FastAPI's Path
from pathlib import Path as PathLib
import speech_recognition as sr
import sounddevice as sd
import librosa
import tensorflow as tf
from loguru import logger
import tempfile
import uuid
import wave
import struct
from scipy.io import wavfile
import json
import traceback
import asyncio

# Set up logger
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
logger.add("willspeak_server.log", rotation="10 MB", level="DEBUG")

# Constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB max upload size
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

app = FastAPI(title="WillSpeak API")

# Create data and models directories if they don't exist
data_dir = PathLib(__file__).parent.parent.parent / "data"
models_dir = PathLib(__file__).parent.parent.parent / "models"
logs_dir = PathLib(__file__).parent.parent.parent / "logs"
temp_dir = PathLib(tempfile.gettempdir()) / "willspeak"

for directory in [data_dir, models_dir, logs_dir, temp_dir]:
    try:
        if not directory.exists():
            os.makedirs(directory)
            logger.info(f"Created directory at {directory}")
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")

# Create user and training directories
user_dir = data_dir / "users"
training_dir = data_dir / "training"
training_data_dir = data_dir / "training_data"

for directory in [user_dir, training_dir, training_data_dir]:
    try:
        if not directory.exists():
            os.makedirs(directory)
            logger.info(f"Created directory at {directory}")
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import speech enhancement model functionality
from willspeak_server.ml.model_integration import get_model, enhance_audio as enhance_audio_with_model, initialize_models

# Custom exception class for specific audio processing errors
class AudioProcessingError(Exception):
    """Exception raised for errors during audio processing."""
    def __init__(self, message, details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)

# Helper Functions
def preprocess_audio(audio_data, sample_rate, target_sr=DEFAULT_SAMPLE_RATE):
    """
    Preprocess audio data:
    - Resample to target sample rate
    - Convert to mono if stereo
    - Normalize amplitude
    - Remove silence

    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Original sample rate
        target_sr: Target sample rate

    Returns:
        Preprocessed audio data
    """
    if audio_data is None:
        raise AudioProcessingError("Audio data is None")

    if len(audio_data) == 0:
        raise AudioProcessingError("Audio data is empty")

    logger.info(f"Preprocessing audio: original sr={sample_rate}, target sr={target_sr}")
    logger.debug(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")

    try:
        # Convert to mono if necessary
        if len(audio_data.shape) > 1:
            logger.info("Converting stereo to mono")
            audio_data = librosa.to_mono(audio_data.T)
    except Exception as e:
        logger.warning(f"Error converting to mono: {e}, attempting alternative method")
        # Alternative method for stereo to mono conversion
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)

    # Ensure audio is float type for processing
    if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
        max_value = np.iinfo(audio_data.dtype).max if np.issubdtype(audio_data.dtype, np.integer) else 1.0
        audio_data = audio_data.astype(np.float32) / max_value
        logger.debug(f"Converted audio to float32, normalized by {max_value}")

    # Resample if necessary
    if sample_rate != target_sr:
        try:
            logger.info(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            raise AudioProcessingError(f"Resampling failed: {e}")

    # Normalize amplitude
    try:
        logger.info("Normalizing amplitude")
        audio_data = librosa.util.normalize(audio_data)
    except Exception as e:
        logger.warning(f"Amplitude normalization failed: {e}, using alternative method")
        # Alternative normalization
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

    # Remove silence
    try:
        logger.info("Removing silence")
        non_silent_intervals = librosa.effects.split(
            audio_data,
            top_db=20,
            frame_length=512,
            hop_length=128
        )

        # If silence removal found intervals, keep only non-silent parts
        if len(non_silent_intervals) > 0:
            audio_parts = []
            for interval in non_silent_intervals:
                start, end = interval
                audio_parts.append(audio_data[start:end])
            audio_data = np.concatenate(audio_parts)
            logger.debug(f"After silence removal, audio length: {len(audio_data)}")
        else:
            logger.warning("No non-silent intervals found, keeping original audio")
    except Exception as e:
        logger.warning(f"Silence removal failed: {e}, keeping original audio")

    # Final check for invalid values
    if np.isnan(audio_data).any() or np.isinf(audio_data).any():
        logger.warning("Found NaN or Inf values in processed audio, replacing with zeros")
        audio_data = np.nan_to_num(audio_data)

    logger.info("Preprocessing complete")
    return audio_data

def extract_features(audio_data, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract features from audio data using librosa

    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for feature extraction

    Returns:
        Dictionary of features and MFCC features array
    """
    logger.info(f"Extracting features: n_mfcc={n_mfcc}, n_fft={n_fft}")

    try:
        # Extract MFCC features
        mfcc_features = librosa.feature.mfcc(
            y=audio_data,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )

        # Extract other features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )[0]

        # Calculate mean and std of features
        mfcc_mean = np.mean(mfcc_features, axis=1)
        mfcc_std = np.std(mfcc_features, axis=1)

        # Combine into a feature dictionary
        features = {
            "mfcc_mean": mfcc_mean.tolist(),
            "mfcc_std": mfcc_std.tolist(),
            "spectral_centroid_mean": float(np.mean(spectral_centroid))
        }

        logger.info("Feature extraction complete")
        return features, mfcc_features
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise AudioProcessingError(f"Feature extraction failed: {e}")

def enhance_speech(audio_data, sample_rate, features=None, user_id=None):
    """
    Enhance speech using the model

    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        features: Optional audio features
        user_id: Optional user ID for personalized enhancement

    Returns:
        Enhanced audio data
    """
    try:
        # Delegate to the model integration module
        enhanced_audio, metrics = enhance_audio_with_model(audio_data, sample_rate, user_id)
        return enhanced_audio
    except Exception as e:
        logger.error(f"Speech enhancement failed: {e}")
        # Return original audio if enhancement fails
        logger.warning("Returning original audio due to enhancement failure")
        return audio_data

def transcribe_audio(audio_data, sample_rate):
    """
    Transcribe audio using SpeechRecognition

    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio

    Returns:
        Transcribed text or empty string on failure
    """
    recognizer = sr.Recognizer()

    try:
        # Convert numpy array to AudioData object
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        audio_data_bytes = audio_data_int16.tobytes()
        audio_source = sr.AudioData(audio_data_bytes, sample_rate, 2)

        try:
            # Use Google's speech recognition service
            text = recognizer.recognize_google(audio_source)
            logger.info(f"Transcription: {text}")
            return text
        except sr.UnknownValueError:
            logger.warning("Speech Recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Could not request results from Speech Recognition service: {e}")
            return ""
    except Exception as e:
        logger.error(f"Transcription preparation failed: {e}")
        return ""

def read_audio_file(file_path):
    """
    Read audio file using multiple methods to ensure robustness

    Args:
        file_path: Path to the audio file

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    file_path_str = str(file_path)
    logger.info(f"Reading audio file: {file_path_str}")

    # Try multiple methods to read the audio file
    methods_tried = []

    # Method 1: Try using soundfile
    try:
        logger.debug(f"Attempting to read with soundfile")
        methods_tried.append("soundfile")
        audio_data, sample_rate = sf.read(file_path_str)
        logger.info(f"Successfully read with soundfile: sr={sample_rate}, shape={audio_data.shape}, dtype={audio_data.dtype}")
        return audio_data, sample_rate
    except Exception as e:
        logger.warning(f"Failed to read with soundfile: {str(e)}")

    # Method 2: Try using scipy.io.wavfile
    try:
        logger.debug(f"Attempting to read with scipy.io.wavfile")
        methods_tried.append("scipy.io.wavfile")
        sample_rate, audio_data = wavfile.read(file_path_str)

        # Convert to float if needed
        if audio_data.dtype in [np.int16, np.int32]:
            max_val = 32768.0 if audio_data.dtype == np.int16 else 2147483648.0
            audio_data = audio_data.astype(np.float32) / max_val

        logger.info(f"Successfully read with scipy.io.wavfile: sr={sample_rate}, shape={audio_data.shape}, dtype={audio_data.dtype}")
        return audio_data, sample_rate
    except Exception as e:
        logger.warning(f"Failed to read with scipy.io.wavfile: {str(e)}")

    # Method 3: Try using wave module
    try:
        logger.debug(f"Attempting to read with wave module")
        methods_tried.append("wave")
        with wave.open(file_path_str, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()

            logger.debug(f"Wave file info: channels={n_channels}, width={sample_width}, rate={sample_rate}, frames={n_frames}")

            # Read all frames
            raw_data = wf.readframes(n_frames)

            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                format_str = f"{n_frames * n_channels}h"
                audio_data = np.array(struct.unpack(format_str, raw_data))
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif sample_width == 4:  # 32-bit
                format_str = f"{n_frames * n_channels}i"
                audio_data = np.array(struct.unpack(format_str, raw_data))
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:  # 8-bit or other
                format_str = f"{n_frames * n_channels}b"
                audio_data = np.array(struct.unpack(format_str, raw_data))
                audio_data = audio_data.astype(np.float32) / 128.0

            # Convert to mono if needed
            if n_channels > 1:
                audio_data = audio_data.reshape(-1, n_channels)
                audio_data = np.mean(audio_data, axis=1)

        logger.info(f"Successfully read with wave module: sr={sample_rate}, length={len(audio_data)}, dtype={audio_data.dtype}")
        return audio_data, sample_rate
    except Exception as e:
        logger.warning(f"Failed to read with wave module: {str(e)}")

    # Method 4: Try using librosa as a last resort
    try:
        logger.debug(f"Attempting to read with librosa")
        methods_tried.append("librosa")
        audio_data, sample_rate = librosa.load(file_path_str, sr=None)
        logger.info(f"Successfully read with librosa: sr={sample_rate}, shape={audio_data.shape}, dtype={audio_data.dtype}")
        return audio_data, sample_rate
    except Exception as e:
        logger.warning(f"Failed to read with librosa: {str(e)}")

    # If all methods fail, raise a detailed exception
    error_msg = f"Failed to read audio file with methods: {', '.join(methods_tried)}"
    logger.error(error_msg)
    raise AudioProcessingError(error_msg)

def save_audio_file(audio_data, sample_rate, filepath):
    """
    Save audio data to file with fallback methods

    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        filepath: Path to save the audio file

    Returns:
        bool: True if successful
    """
    filepath_str = str(filepath)
    logger.info(f"Saving audio to: {filepath_str}")

    # Make sure directory exists
    os.makedirs(os.path.dirname(filepath_str), exist_ok=True)

    # Try multiple methods to save the audio
    methods_tried = []

    # Method 1: Try using soundfile
    try:
        logger.debug(f"Attempting to save with soundfile")
        methods_tried.append("soundfile")
        sf.write(filepath_str, audio_data, sample_rate)
        logger.info(f"Successfully saved with soundfile")
        return True
    except Exception as e:
        logger.warning(f"Failed to save with soundfile: {str(e)}")

    # Method 2: Try using scipy.io.wavfile
    try:
        logger.debug(f"Attempting to save with scipy.io.wavfile")
        methods_tried.append("scipy.io.wavfile")
        # Convert to int16 for saving
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(filepath_str, sample_rate, audio_data_int16)
        logger.info(f"Successfully saved with scipy.io.wavfile")
        return True
    except Exception as e:
        logger.warning(f"Failed to save with scipy.io.wavfile: {str(e)}")

    # Method 3: Try using wave module
    try:
        logger.debug(f"Attempting to save with wave module")
        methods_tried.append("wave")
        # Convert to int16 for saving
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(filepath_str, 'wb') as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data_int16.tobytes())

        logger.info(f"Successfully saved with wave module")
        return True
    except Exception as e:
        logger.warning(f"Failed to save with wave module: {str(e)}")

    # If all methods fail, raise a detailed exception
    error_msg = f"Failed to save audio file with methods: {', '.join(methods_tried)}"
    logger.error(error_msg)
    raise AudioProcessingError(error_msg)

# API rate limiting logic
rate_limit = {}

def check_rate_limit(client_id: str, limit: int = 10, window: int = 60) -> bool:
    """
    Check if a client has exceeded the rate limit

    Args:
        client_id: Identifier for the client (e.g., IP address)
        limit: Maximum requests allowed in the window
        window: Time window in seconds

    Returns:
        bool: True if rate limit is not exceeded, False otherwise
    """
    current_time = asyncio.get_event_loop().time()

    if client_id not in rate_limit:
        rate_limit[client_id] = []

    # Remove timestamps outside the window
    rate_limit[client_id] = [t for t in rate_limit[client_id] if current_time - t < window]

    # Check if limit is exceeded
    if len(rate_limit[client_id]) >= limit:
        return False

    # Add current timestamp
    rate_limit[client_id].append(current_time)
    return True

# Dependency for rate limiting
async def rate_limiter(client_id: str = Depends(lambda: "default")):
    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return client_id

# Endpoints
@app.get("/", dependencies=[Depends(rate_limiter)])
async def root():
    """Root endpoint - health check"""
    return {"status": "ok", "message": "WillSpeak API is running"}

async def process_audio_generic(file: UploadFile, user_id=None, client_id=None):
    """
    Generic audio processing function that can be called with or without a user ID

    Args:
        file: Uploaded audio file
        user_id: Optional user ID for personalized processing
        client_id: Optional client ID for rate limiting

    Returns:
        Dictionary with processing results
    """
    # Apply rate limiting
    if client_id and not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    logger.info(f"Processing audio file: {file.filename} for user: {user_id or 'default'}")

    temp_path = None
    try:
        # Validate file format by extension
        if file.filename:
            file_ext = PathLib(file.filename).suffix.lower()
            if file_ext not in SUPPORTED_AUDIO_FORMATS:
                logger.warning(f"Unsupported file format: {file_ext}")
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Unsupported file format: {file_ext}. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"}
                )

        # Read file content
        content = await file.read()

        # Check file size
        if len(content) > MAX_UPLOAD_SIZE:
            logger.warning(f"File too large: {len(content)} bytes")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"File too large. Maximum size: {MAX_UPLOAD_SIZE/1024/1024} MB"}
            )

        # Create a unique temp filename
        temp_filename = f"willspeak_temp_{uuid.uuid4()}{PathLib(file.filename).suffix}"
        temp_path = temp_dir / temp_filename

        # Save uploaded content to temp file
        os.makedirs(temp_dir, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(content)

        # Try to read the audio file
        try:
            audio_data, sample_rate = read_audio_file(temp_path)
        except AudioProcessingError as e:
            logger.error(f"Failed to read audio file: {e}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Failed to read audio file: {str(e)}"}
            )

        # Check if audio_data is valid
        if audio_data is None or len(audio_data) == 0:
            logger.error("Audio data is empty")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Audio data is empty or could not be read"}
            )

        logger.info(f"Audio file loaded: {len(audio_data)} samples, {sample_rate}Hz")

        # Preprocess audio
        try:
            preprocessed_audio = preprocess_audio(audio_data, sample_rate)
        except AudioProcessingError as e:
            logger.error(f"Preprocessing failed: {e}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Audio preprocessing failed: {str(e)}"}
            )

        # Extract features
        try:
            features, mfcc_features = extract_features(preprocessed_audio, DEFAULT_SAMPLE_RATE)
        except AudioProcessingError as e:
            logger.error(f"Feature extraction failed: {e}")
            features = {"error": str(e)}
            mfcc_features = None

        # Try to transcribe the audio
        transcription = transcribe_audio(preprocessed_audio, DEFAULT_SAMPLE_RATE)

        # Enhance speech with the user's model if provided
        try:
            enhanced_audio = enhance_speech(preprocessed_audio, DEFAULT_SAMPLE_RATE, user_id=user_id)
        except Exception as e:
            logger.error(f"Speech enhancement failed: {e}")
            enhanced_audio = preprocessed_audio  # Use preprocessed audio if enhancement fails

        # Generate a unique filename for the processed audio
        output_filename = f"processed_{uuid.uuid4()}.wav"
        output_path = data_dir / output_filename

        # Save processed file
        try:
            success = save_audio_file(enhanced_audio, DEFAULT_SAMPLE_RATE, output_path)
            if not success:
                logger.warning("Failed to save processed audio, using alternative method")
                # Alternative method
                sf.write(str(output_path), enhanced_audio, DEFAULT_SAMPLE_RATE)
        except Exception as e:
            logger.error(f"Failed to save processed audio: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Failed to save processed audio: {str(e)}"}
            )

        return {
            "status": "success",
            "message": "Audio processed successfully",
            "transcription": transcription,
            "features": features,
            "output_file": output_filename
        }
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")

@app.post("/process-audio", dependencies=[Depends(rate_limiter)])
async def process_audio(file: UploadFile = File(...), client_id: str = Depends(rate_limiter)):
    """
    Process audio file and return enhanced version

    Args:
        file: Uploaded audio file
        client_id: Client ID for rate limiting

    Returns:
        Processing results
    """
    return await process_audio_generic(file, client_id=client_id)

@app.post("/process-audio-for-user")
async def process_audio_for_user(user_id: str = Form(...), file: UploadFile = File(...), client_id: str = Depends(rate_limiter)):
    """
    Process audio with user-specific model

    Args:
        user_id: User ID for personalized processing
        file: Uploaded audio file
        client_id: Client ID for rate limiting

    Returns:
        Processing results
    """
    from willspeak_server.api.endpoints import verify_user_exists

    # Check if user exists
    user_exists = await verify_user_exists(user_id)
    if not user_exists:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"User not found: {user_id}"}
        )

    # Process audio with user's model
    return await process_audio_generic(file, user_id=user_id, client_id=client_id)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio processing
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    # Get user ID from query params if provided
    user_id = None
    if 'user_id' in websocket.query_params:
        user_id = websocket.query_params['user_id']
        logger.info(f"WebSocket connection for user: {user_id}")

    # For rate limiting
    client_id = f"ws_{websocket.client.host}"
    request_times = []

    try:
        while True:
            # Apply simple rate limiting
            current_time = asyncio.get_event_loop().time()
            request_times = [t for t in request_times if current_time - t < 60]  # 1-minute window
            if len(request_times) >= 100:  # 100 requests per minute max
                logger.warning(f"WebSocket rate limit exceeded for {client_id}")
                await websocket.send_text(json.dumps({"error": "Rate limit exceeded"}))
                await asyncio.sleep(1)  # Wait before trying again
                continue

            request_times.append(current_time)

            # Receive audio data
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.debug("WebSocket receive timeout - sending keepalive")
                await websocket.send_text(json.dumps({"status": "keepalive"}))
                continue

            try:
                # Convert bytes to numpy array (assuming 16-bit PCM format)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0

                # Check for invalid audio data
                if len(audio_chunk) == 0 or np.isnan(audio_chunk).any() or np.isinf(audio_chunk).any():
                    logger.warning("Received invalid audio chunk")
                    # Send original data back
                    await websocket.send_bytes(data)
                    continue

                # Process audio
                processed_chunk = preprocess_audio(audio_chunk, DEFAULT_SAMPLE_RATE, DEFAULT_SAMPLE_RATE)

                # Enhance audio
                enhanced_chunk = enhance_speech(processed_chunk, DEFAULT_SAMPLE_RATE, user_id=user_id)

                # Convert back to bytes
                enhanced_bytes = (enhanced_chunk * 32767).astype(np.int16).tobytes()

                # Send enhanced audio back
                await websocket.send_bytes(enhanced_bytes)
            except Exception as e:
                logger.error(f"Error processing audio chunk: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                # Send original data back on error
                try:
                    await websocket.send_bytes(data)
                except Exception as ws_error:
                    logger.error(f"WebSocket send error: {ws_error}")
                    break  # Break out of loop if we can't send data
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("WebSocket connection closed")

# Add debug endpoint for testing
@app.get("/debug/audio-test")
async def debug_audio_test(sample_rate: int = 16000):
    """
    Generate a test audio file for debugging

    Args:
        sample_rate: Sample rate to use

    Returns:
        Path to the test file
    """
    try:
        # Generate a simple sine wave
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Save to file
        output_filename = f"test_audio_{uuid.uuid4()}.wav"
        output_path = data_dir / output_filename
        save_audio_file(audio_data, sample_rate, output_path)

        return {
            "status": "success",
            "message": "Test audio generated",
            "file_path": str(output_path),
            "sample_rate": sample_rate,
            "duration": duration
        }
    except Exception as e:
        logger.error(f"Error creating test audio: {str(e)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(e)}
    )

    # Import training data router first


from willspeak_server.api.training_data_endpoints import training_data_router

# Add the training data router to the app
app.include_router(training_data_router)
logger.info("Included training data router from training_data_endpoints.py")

# Import user and training endpoints
from willspeak_server.api.endpoints import router

app.include_router(router)
logger.info("Included router from endpoints.py")

# Log all registered routes
for route in app.routes:
    if isinstance(route, APIRoute):
        logger.info(f"Route: {route.path} - {route.methods}")


# Exception handlers
@app.exception_handler(AudioProcessingError)
async def audio_processing_exception_handler(request, exc):
    logger.error(f"AudioProcessingError: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": exc.message, "details": exc.details}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.debug(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error", "details": str(exc)}
    )


# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    try:
        # Initialize models
        initialize_models()
        logger.info("Models initialized on startup")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")


# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    try:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {file}: {e}")
        logger.info("Cleanup completed on shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")


# Main block for direct execution
if __name__ == "__main__":
    # Initialize models
    initialize_models()

    import uvicorn

    logger.info("Starting WillSpeak Server directly")
    uvicorn.run(app, host="127.0.0.1", port=8000)