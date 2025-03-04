"""
Model integration for WillSpeak server

Integrates the speech enhancement model with the FastAPI server
"""
import os
import sys
from pathlib import Path
from loguru import logger
import numpy as np
import soundfile as sf
import io
import json
import tensorflow as tf

# Add the parent directory to the path so we can import our modules
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from willspeak_server.ml.speech_enhancement_model import SpeechEnhancementModel

# Global model instance
_global_model = None
_user_models = {}


def get_model(user_id=None):
    """
    Get or create a speech enhancement model.

    Args:
        user_id (str, optional): User ID for personalized model

    Returns:
        SpeechEnhancementModel: Model instance
    """
    global _global_model, _user_models

    # If user_id is provided, try to get user-specific model
    if user_id:
        if user_id in _user_models:
            return _user_models[user_id]

        # Create new user model
        model = SpeechEnhancementModel(user_id=user_id)
        if model.load_model(user_specific=True):
            _user_models[user_id] = model
            return model

    # Fall back to global model
    if _global_model is None:
        _global_model = SpeechEnhancementModel()
        _global_model.load_model(user_specific=False)

    return _global_model


def initialize_models():
    """
    Initialize speech enhancement models.
    """
    global _global_model

    logger.info("Initializing speech enhancement models")

    # Initialize global model
    _global_model = SpeechEnhancementModel()

    # Try to load pre-trained model
    if not _global_model.load_model(user_specific=False):
        logger.warning("No pre-trained model found. Building new model.")
        _global_model.build_model()

    logger.info("Speech enhancement models initialized")


def enhance_audio(audio_data, sample_rate, user_id=None):
    """
    Enhance speech audio.

    Args:
        audio_data (np.array): Audio data to enhance
        sample_rate (int): Sample rate of audio
        user_id (str, optional): User ID for personalized model

    Returns:
        np.array: Enhanced audio
    """
    model = get_model(user_id)

    # If no model is available, return original audio
    if model.model is None:
        logger.warning("No model available for enhancement, returning original audio")
        return audio_data

    # Enhance audio
    logger.info(f"Enhancing audio with model for user: {user_id or 'default'}")
    enhanced_audio = model.enhance_audio(audio_data, sample_rate)

    return enhanced_audio


def train_model(unclear_audio_paths, clear_audio_paths, user_id, epochs=50, batch_size=32):
    """
    Train speech enhancement model.

    Args:
        unclear_audio_paths (list): Paths to unclear speech audio files
        clear_audio_paths (list): Paths to clear speech audio files
        user_id (str): User ID for personalized model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training

    Returns:
        dict: Training metrics
    """
    model = get_model(user_id) or SpeechEnhancementModel(user_id=user_id)

    try:
        # Prepare training data
        train_data = model.prepare_training_data(unclear_audio_paths, clear_audio_paths)

        # Split into train/test
        test_size = max(1, int(len(unclear_audio_paths) * 0.2))
        test_data = (train_data[0][-test_size:], train_data[1][-test_size:])
        train_data = (train_data[0][:-test_size], train_data[1][:-test_size])

        # Train model
        logger.info(f"Training model for user: {user_id}")
        history = model.train(train_data, test_data, epochs=epochs, batch_size=batch_size)

        # Save trained model
        model.save_model(user_specific=True)

        # Update global dictionary
        _user_models[user_id] = model

        # Get final metrics
        metrics = {
            'loss': float(history['loss'][-1]),
            'val_loss': float(history['val_loss'][-1]) if 'val_loss' in history else None,
            'mae': float(history['mean_absolute_error'][-1]),
            'val_mae': float(history['val_mean_absolute_error'][-1]) if 'val_mean_absolute_error' in history else None,
            'epochs_completed': len(history['loss']),
            'training_samples': len(train_data[0])
        }

        logger.info(f"Training completed for user {user_id}")
        return metrics

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise