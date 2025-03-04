"""
Audio preprocessing functions for WillSpeak
"""
import numpy as np
import librosa
from willspeak_server.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_audio(audio_data, sample_rate, target_sr=16000):
    """
    Preprocess audio data:
    - Resample to target sample rate
    - Convert to mono if stereo
    - Normalize amplitude
    - Remove silence
    - Apply noise reduction

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

    # Simple noise reduction (placeholder for more sophisticated methods)
    # In a real application, we'd use more advanced techniques

    logger.info("Preprocessing complete")
    return audio_data