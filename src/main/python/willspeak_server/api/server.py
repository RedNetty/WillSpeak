"""
WillSpeak Server - FastAPI Application
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import soundfile as sf
from pathlib import Path
import os
import sys
import speech_recognition as sr
import sounddevice as sd
import librosa
import tensorflow as tf
from loguru import logger

# Set up logger
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
logger.add("willspeak_server.log", rotation="10 MB", level="DEBUG")

app = FastAPI(title="WillSpeak API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directory if it doesn't exist
data_dir = Path(__file__).parent.parent.parent / "data"
models_dir = Path(__file__).parent.parent.parent / "models"

for directory in [data_dir, models_dir]:
    if not directory.exists():
        os.makedirs(directory)
        logger.info(f"Created directory at {directory}")

# Import speech enhancement model functionality
from willspeak_server.ml.model_integration import get_model, enhance_audio as enhance_audio_with_model

def preprocess_audio(audio_data, sample_rate, target_sr=16000):
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
    logger.info(f"Preprocessing audio: original sr={sample_rate}, target sr={target_sr}")

    # Convert to mono if necessary
    if len(audio_data.shape) > 1:
        logger.info("Converting stereo to mono")
        audio_data = librosa.to_mono(audio_data.T)

    # Resample if necessary
    if sample_rate != target_sr:
        logger.info(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)

    # Normalize amplitude
    logger.info("Normalizing amplitude")
    audio_data = librosa.util.normalize(audio_data)

    # Remove silence
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

    logger.info("Preprocessing complete")
    return audio_data

def extract_features(audio_data, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract features from audio data using librosa
    """
    logger.info(f"Extracting features: n_mfcc={n_mfcc}, n_fft={n_fft}")

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

def enhance_speech(audio_data, sample_rate, features=None, user_id=None):
    """
    Enhance speech using the model
    """
    # Delegate to the model integration module
    return enhance_audio_with_model(audio_data, sample_rate, user_id)

def transcribe_audio(audio_data, sample_rate):
    """
    Transcribe audio using SpeechRecognition
    """
    recognizer = sr.Recognizer()

    # Convert numpy array to AudioData object
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    audio_data_bytes = audio_data_int16.tobytes()
    audio_source = sr.AudioData(audio_data_bytes, sample_rate, 2)

    try:
        # Try Google's speech recognition service
        text = recognizer.recognize_google(audio_source)
        logger.info(f"Transcription: {text}")
        return text
    except sr.UnknownValueError:
        logger.warning("Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        logger.error(f"Could not request results from Speech Recognition service: {e}")
        return ""

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {"status": "ok", "message": "WillSpeak API is running"}

async def process_audio_generic(file: UploadFile, user_id=None):
    """
    Generic audio processing function that can be called with or without a user ID
    """
    logger.info(f"Processing audio file: {file.filename} for user: {user_id or 'default'}")

    try:
        # Read audio file content
        content = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(content))

        # Preprocess audio
        preprocessed_audio = preprocess_audio(audio_data, sample_rate)

        # Extract features
        features, mfcc_features = extract_features(preprocessed_audio, sample_rate)

        # Try to transcribe the audio (using SpeechRecognition)
        transcription = transcribe_audio(preprocessed_audio, 16000)

        # Enhance speech with the user's model if provided
        enhanced_audio = enhance_speech(preprocessed_audio, 16000, user_id=user_id)

        # Save processed file
        output_filename = f"processed_{file.filename}"
        output_path = data_dir / output_filename
        sf.write(output_path, enhanced_audio, 16000)

        return {
            "status": "success",
            "message": "Audio processed successfully",
            "transcription": transcription,
            "features": features,
            "output_file": output_filename
        }

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Process audio file and return enhanced version
    """
    return await process_audio_generic(file)

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

    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()

            try:
                # Convert bytes to numpy array (assuming 16-bit PCM format)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0

                # Process audio (basic version)
                processed_chunk = preprocess_audio(audio_chunk, 16000, 16000)

                # Enhance audio (when model is available)
                enhanced_chunk = enhance_speech(processed_chunk, 16000, user_id=user_id)

                # Convert back to bytes
                enhanced_bytes = (enhanced_chunk * 32767).astype(np.int16).tobytes()

                # Send enhanced audio back
                await websocket.send_bytes(enhanced_bytes)

            except Exception as e:
                logger.error(f"Error processing audio chunk: {str(e)}")
                # Send original data back on error
                await websocket.send_bytes(data)

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

    finally:
        logger.info("WebSocket connection closed")

# Import and include the endpoints router
from willspeak_server.api.endpoints import router
app.include_router(router)

if __name__ == "__main__":
    # Initialize models
    from willspeak_server.ml.model_integration import initialize_models
    initialize_models()

    import uvicorn
    logger.info("Starting WillSpeak Server directly")
    uvicorn.run(app, host="127.0.0.1", port=8000)