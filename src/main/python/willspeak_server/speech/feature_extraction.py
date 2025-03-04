"""
Audio feature extraction for WillSpeak
"""
import numpy as np
import librosa
from python_speech_features import mfcc
from willspeak_server.utils.logger import get_logger

logger = get_logger(__name__)

def extract_features(audio_data, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract features from audio data:
    - MFCC features
    - Spectral features
    - Prosodic features

    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for feature extraction

    Returns:
        Dictionary of features
    """
    logger.info(f"Extracting features: n_mfcc={n_mfcc}, n_fft={n_fft}")

    # Extract MFCC features
    mfcc_features = mfcc(
        audio_data,
        samplerate=sample_rate,
        numcep=n_mfcc,
        nfilt=26,
        nfft=n_fft
    )

    # Calculate mean and std of MFCC features
    mfcc_mean = np.mean(mfcc_features, axis=0)
    mfcc_std = np.std(mfcc_features, axis=0)

    # Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    # Extract prosodic features
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_data,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate
    )

    # Replace NaN with 0 for unvoiced segments
    f0 = np.nan_to_num(f0)

    # Compute statistics for pitch
    pitch_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0
    pitch_std = np.std(f0[f0 > 0]) if np.any(f0 > 0) else 0

    # Compute envelope and energy
    envelope = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
    energy = np.sum(envelope**2, axis=0)

    # Assemble features dictionary
    features = {
        "mfcc_features": mfcc_features,
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "energy_mean": np.mean(energy),
        "energy_std": np.std(energy)
    }

    logger.info("Feature extraction complete")
    return features