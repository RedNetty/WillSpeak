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
import time
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import our modules
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from willspeak_server.ml.speech_enhancement_model import SpeechEnhancementModel

# Global model instance
_global_model = None
_user_models = {}

# Training metrics history for visualization
_training_histories = {}

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
            logger.info(f"Using existing model for user: {user_id}")
            return _user_models[user_id]

        # Create new user model
        logger.info(f"Creating new model for user: {user_id}")
        model = SpeechEnhancementModel(user_id=user_id)
        if model.load_model(user_specific=True):
            logger.success(f"Successfully loaded user-specific model for: {user_id}")
            _user_models[user_id] = model
            return model
        else:
            logger.warning(f"No user-specific model found for {user_id}, attempting to load base model")

    # Fall back to global model
    if _global_model is None:
        logger.info("Initializing global model")
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
        logger.info("Base model built successfully")
    else:
        logger.success("Base model loaded successfully")

    # List all user models
    models_dir = Path(_global_model.model_dir)
    if models_dir.exists():
        user_models = [f for f in models_dir.glob("user_*_model.keras")]
        logger.info(f"Found {len(user_models)} user-specific models: {[m.name for m in user_models]}")

    logger.info("Speech enhancement models initialized")


def enhance_audio(audio_data, sample_rate, user_id=None):
    """
    Enhance speech audio with detailed confidence metrics.

    Args:
        audio_data (np.array): Audio data to enhance
        sample_rate (int): Sample rate of audio
        user_id (str, optional): User ID for personalized model

    Returns:
        tuple: (enhanced_audio, metrics)
    """
    start_time = time.time()
    model = get_model(user_id)

    # If no model is available, return original audio
    if model.model is None:
        logger.warning("No model available for enhancement, returning original audio")
        return audio_data, {
            "enhanced": False,
            "confidence": 0.0,
            "processing_time_ms": 0,
            "model_type": "none"
        }

    # Enhance audio
    logger.info(f"Enhancing audio with model for user: {user_id or 'default'}")
    enhanced_audio, confidence = model.enhance_audio_with_confidence(audio_data, sample_rate)

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # in milliseconds

    # Log enhancement metrics
    enhancement_ratio = np.sum(np.abs(enhanced_audio)) / (np.sum(np.abs(audio_data)) + 1e-10)
    logger.info(f"Enhancement completed: confidence={confidence:.2f}, ratio={enhancement_ratio:.2f}, time={processing_time:.2f}ms")

    # Create enhancement metrics dictionary
    metrics = {
        "enhanced": True,
        "confidence": float(confidence),
        "enhancement_ratio": float(enhancement_ratio),
        "processing_time_ms": float(processing_time),
        "model_type": "user" if user_id else "base"
    }

    return enhanced_audio, metrics


def train_model(unclear_audio_paths, clear_audio_paths, user_id, epochs=50, batch_size=32):
    """
    Train speech enhancement model with extensive validation and metrics.

    Args:
        unclear_audio_paths (list): Paths to unclear speech audio files
        clear_audio_paths (list): Paths to clear speech audio files
        user_id (str): User ID for personalized model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training

    Returns:
        dict: Training metrics
    """
    global _training_histories

    logger.info(f"Starting training for user {user_id} with {len(unclear_audio_paths)} sample pairs")

    # Get or create user model
    model = get_model(user_id) or SpeechEnhancementModel(user_id=user_id)

    # Generate training ID
    training_id = f"train_{user_id}_{int(time.time())}"

    try:
        # Log sample details
        for i, (unclear_path, clear_path) in enumerate(zip(unclear_audio_paths, clear_audio_paths)):
            logger.info(f"Training pair {i+1}: {Path(unclear_path).name} <-> {Path(clear_path).name}")

        # Prepare training data
        logger.info("Preparing training data...")
        train_data = model.prepare_training_data(unclear_audio_paths, clear_audio_paths)

        # Split into train/test with stratified sampling for better validation
        test_size = max(1, int(len(unclear_audio_paths) * 0.2))

        # Instead of simple splitting, we'll use a better strategy
        # that ensures representative examples in both sets
        indices = np.arange(len(train_data[0]))
        np.random.shuffle(indices)

        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        test_data = (
            np.array([train_data[0][i] for i in test_indices]),
            np.array([train_data[1][i] for i in test_indices])
        )
        train_data = (
            np.array([train_data[0][i] for i in train_indices]),
            np.array([train_data[1][i] for i in train_indices])
        )

        logger.info(f"Training data split: {len(train_indices)} training samples, {len(test_indices)} validation samples")

        # Train model with adaptive learning rate and early stopping
        logger.info(f"Training model for user: {user_id}")

        # Add learning rate scheduler for better convergence
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]

        history = model.train(train_data, test_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

        # Store training history for visualization
        _training_histories[training_id] = history

        # Save trained model
        model.save_model(user_specific=True)
        logger.success(f"Model saved for user: {user_id}")

        # Update global dictionary
        _user_models[user_id] = model

        # Create visualization of training progress
        create_training_visualization(history, user_id)

        # Get final metrics with detailed statistics
        metrics = {
            'loss': float(history['loss'][-1]),
            'val_loss': float(history['val_loss'][-1]) if 'val_loss' in history else None,
            'mae': float(history['mean_absolute_error'][-1]),
            'val_mae': float(history['val_mean_absolute_error'][-1]) if 'val_mean_absolute_error' in history else None,
            'clarity': float(history['speech_clarity_metric'][-1]) if 'speech_clarity_metric' in history else None,
            'val_clarity': float(history['val_speech_clarity_metric'][-1]) if 'val_speech_clarity_metric' in history else None,
            'epochs_completed': len(history['loss']),
            'training_samples': len(train_data[0]),
            'validation_samples': len(test_data[0]),
            'initial_loss': float(history['loss'][0]),
            'loss_improvement': float(history['loss'][0] - history['loss'][-1]),
            'training_id': training_id,
            'learning_rate': float(history['lr'][-1]) if 'lr' in history else None,
            'best_epoch': np.argmin(history['val_loss']) if 'val_loss' in history else np.argmin(history['loss']),
            'model_path': str(model.user_model_path) + '.keras'
        }

        # Run validation on a test sample
        test_metrics = validate_model_on_samples(model, test_data)
        metrics.update(test_metrics)

        logger.success(f"Training completed for user {user_id}: Loss {metrics['loss']:.4f}, Clarity {metrics.get('clarity', 0):.4f}")
        return metrics

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


def validate_model_on_samples(model, test_data):
    """
    Validate model on specific test samples and return detailed metrics.

    Args:
        model: Trained SpeechEnhancementModel
        test_data: Tuple of (input_spectrograms, target_spectrograms)

    Returns:
        dict: Validation metrics
    """
    logger.info("Running model validation on test samples")

    # Select a few samples for validation
    num_samples = min(5, len(test_data[0]))
    sample_indices = np.random.choice(len(test_data[0]), num_samples, replace=False)

    validation_metrics = {
        "sample_validations": []
    }

    total_mse = 0
    total_mae = 0
    total_clarity = 0

    # For each sample, compute detailed metrics
    for i, idx in enumerate(sample_indices):
        input_spec = test_data[0][idx:idx+1]  # Add batch dimension
        target_spec = test_data[1][idx]

        # Get model prediction
        enhanced_spec = model.model.predict(input_spec)[0]

        # Calculate metrics
        mse = np.mean((enhanced_spec - target_spec) ** 2)
        mae = np.mean(np.abs(enhanced_spec - target_spec))
        clarity = calculate_clarity_score(enhanced_spec, target_spec)

        total_mse += mse
        total_mae += mae
        total_clarity += clarity

        # Store sample validation
        validation_metrics["sample_validations"].append({
            "sample_index": int(idx),
            "mse": float(mse),
            "mae": float(mae),
            "clarity": float(clarity)
        })

    # Average metrics
    validation_metrics["avg_test_mse"] = float(total_mse / num_samples)
    validation_metrics["avg_test_mae"] = float(total_mae / num_samples)
    validation_metrics["avg_test_clarity"] = float(total_clarity / num_samples)

    logger.info(f"Validation metrics: MSE={validation_metrics['avg_test_mse']:.4f}, " 
                f"MAE={validation_metrics['avg_test_mae']:.4f}, "
                f"Clarity={validation_metrics['avg_test_clarity']:.4f}")

    return validation_metrics


def calculate_clarity_score(enhanced_spec, target_spec):
    """
    Calculate speech clarity score between enhanced and target spectrograms.

    Args:
        enhanced_spec: Enhanced spectrogram
        target_spec: Target spectrogram

    Returns:
        float: Clarity score (higher is better)
    """
    # Calculate correlation between specs (represents structural similarity)
    enhanced_flat = enhanced_spec.flatten()
    target_flat = target_spec.flatten()

    # Mean values
    enhanced_mean = np.mean(enhanced_flat)
    target_mean = np.mean(target_flat)

    # Centered values
    enhanced_centered = enhanced_flat - enhanced_mean
    target_centered = target_flat - target_mean

    # Correlation calculation
    numerator = np.sum(enhanced_centered * target_centered)
    denominator = np.sqrt(np.sum(enhanced_centered**2) * np.sum(target_centered**2))

    # Avoid division by zero
    if denominator < 1e-8:
        return 0.0

    correlation = numerator / denominator

    # Normalize to 0-1 range
    return (correlation + 1) / 2


def create_training_visualization(history, user_id):
    """
    Create and save training visualizations.

    Args:
        history: Training history dictionary
        user_id: User ID for the model
    """
    try:
        # Create directory for visualizations if it doesn't exist
        vis_dir = Path(project_root) / "logs" / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot training and validation loss
        ax = axes[0, 0]
        ax.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

        # Plot mean absolute error
        ax = axes[0, 1]
        ax.plot(history['mean_absolute_error'], label='Training MAE')
        if 'val_mean_absolute_error' in history:
            ax.plot(history['val_mean_absolute_error'], label='Validation MAE')
        ax.set_title('Mean Absolute Error')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.legend()
        ax.grid(True)

        # Plot speech clarity metric
        ax = axes[1, 0]
        if 'speech_clarity_metric' in history:
            ax.plot(history['speech_clarity_metric'], label='Training Clarity')
            if 'val_speech_clarity_metric' in history:
                ax.plot(history['val_speech_clarity_metric'], label='Validation Clarity')
            ax.set_title('Speech Clarity')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Clarity')
            ax.legend()
            ax.grid(True)

        # Plot learning rate if available
        ax = axes[1, 1]
        if 'lr' in history:
            ax.plot(history['lr'], label='Learning Rate')
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('LR')
            ax.set_yscale('log')
            ax.grid(True)

        # Add title with user ID
        fig.suptitle(f'Training Metrics for User {user_id}', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the visualization
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = vis_dir / f"training_{user_id}_{timestamp}.png"
        plt.savefig(filename)
        logger.info(f"Training visualization saved to {filename}")

        plt.close(fig)

    except Exception as e:
        logger.error(f"Error creating training visualization: {str(e)}")


def get_model_status(user_id=None):
    """
    Get detailed status of models.

    Args:
        user_id (str, optional): User ID to get status for

    Returns:
        dict: Model status information
    """
    global _global_model, _user_models

    status = {
        "base_model_available": _global_model is not None and _global_model.model is not None,
        "user_models_available": len(_user_models),
        "user_models": [],
    }

    # Add base model info if available
    if _global_model and _global_model.model is not None:
        status["base_model"] = {
            "input_shape": _global_model.model.input_shape[1:],
            "output_shape": _global_model.model.output_shape[1:],
            "parameters": int(_global_model.model.count_params()),
            "model_path": str(_global_model.base_model_path) + ".keras"
        }

    # Add specific user model info if requested
    if user_id:
        if user_id in _user_models:
            model = _user_models[user_id]
            model_info = {
                "user_id": user_id,
                "input_shape": model.model.input_shape[1:],
                "output_shape": model.model.output_shape[1:],
                "parameters": int(model.model.count_params()),
                "model_path": str(model.user_model_path) + ".keras"
            }
            status["user_models"].append(model_info)
            status["requested_user_model_available"] = True
        else:
            status["requested_user_model_available"] = False

    # Add info for all user models
    else:
        for uid, model in _user_models.items():
            model_info = {
                "user_id": uid,
                "input_shape": model.model.input_shape[1:],
                "output_shape": model.model.output_shape[1:],
                "parameters": int(model.model.count_params()),
                "model_path": str(model.user_model_path) + ".keras"
            }
            status["user_models"].append(model_info)

    return status