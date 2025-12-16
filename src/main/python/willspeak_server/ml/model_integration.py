"""
Improved Model integration for WillSpeak server

Enhanced integration of speech enhancement models with the FastAPI server
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
import shutil
import threading
import concurrent.futures

# Add the parent directory to the path so we can import our modules
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from willspeak_server.ml.speech_enhancement_model import SpeechEnhancementModel
from willspeak_server.speech.preprocessing import preprocess_audio
from willspeak_server.speech.feature_extraction import extract_features, extract_speech_comprehensibility_score

# Global model cache
_global_model = None
_user_models = {}
_model_lock = threading.RLock()  # For thread-safe model access

# Training metrics history for visualization
_training_histories = {}

# Training job status tracking
_training_jobs = {}

def get_model(user_id=None):
    """
    Get or create a speech enhancement model with thread-safe access.

    Args:
        user_id (str, optional): User ID for personalized model

    Returns:
        SpeechEnhancementModel: Model instance
    """
    global _global_model, _user_models

    with _model_lock:
        # If user_id is provided, try to get user-specific model
        if user_id:
            if user_id in _user_models:
                logger.info(f"Using existing model for user: {user_id}")
                return _user_models[user_id]

            # Create new user model and try to load it
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
    Initialize speech enhancement models with improved error handling.
    """
    global _global_model

    logger.info("Initializing speech enhancement models")

    try:
        # Initialize global model
        _global_model = SpeechEnhancementModel()

        # Try to load pre-trained model
        if not _global_model.load_model(user_specific=False):
            logger.warning("No pre-trained model found. Building new model.")
            _global_model.build_model()
            # Save the newly built model
            _global_model.save_model(user_specific=False)
            logger.info("Base model built and saved successfully")
        else:
            logger.success("Base model loaded successfully")

        # List all user models
        models_dir = Path(_global_model.model_dir)
        if models_dir.exists():
            # Look for user model files (both .keras and .h5 formats)
            user_models = [f for f in models_dir.glob("user_*_model*")
                         if f.name.endswith('.keras') or f.name.endswith('.h5')]

            # Extract user IDs from filenames
            user_ids = set()
            for model_path in user_models:
                # Extract user ID from filename (between "user_" and "_model")
                try:
                    filename = model_path.name
                    prefix_end = filename.find('_model')
                    if prefix_end > 5:  # "user_" is 5 chars
                        user_id = filename[5:prefix_end]
                        user_ids.add(user_id)
                except:
                    continue

            logger.info(f"Found {len(user_ids)} user-specific models: {list(user_ids)}")

            # Pre-load some common user models to speed up first access
            for user_id in list(user_ids)[:3]:  # Load up to 3 models to conserve memory
                try:
                    model = SpeechEnhancementModel(user_id=user_id)
                    if model.load_model(user_specific=True):
                        _user_models[user_id] = model
                        logger.info(f"Pre-loaded model for user: {user_id}")
                except Exception as e:
                    logger.error(f"Error pre-loading model for user {user_id}: {e}")
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        # Create a simple fallback model to prevent complete failure
        try:
            _global_model = SpeechEnhancementModel()
            _global_model.build_model()
            logger.warning("Created fallback model due to initialization error")
        except:
            logger.critical("Fatal error: Could not create fallback model")

    logger.info("Speech enhancement models initialization completed")


def enhance_audio(audio_data, sample_rate, user_id=None):
    """
    Enhance speech audio with detailed confidence metrics using the improved model.

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
    if model is None or model.model is None:
        logger.warning("No model available for enhancement, returning original audio")
        return audio_data, {
            "enhanced": False,
            "confidence": 0.0,
            "processing_time_ms": 0,
            "model_type": "none"
        }

    # Enhance audio with comprehensive metrics
    logger.info(f"Enhancing audio with model for user: {user_id or 'default'}")
    enhanced_audio, confidence = model.enhance_audio_with_confidence(audio_data, sample_rate)

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # in milliseconds

    # Calculate enhancement effectiveness metrics
    # 1. Spectral clarity difference
    original_features = extract_features(audio_data, sample_rate)
    enhanced_features = extract_features(enhanced_audio, sample_rate)

    # Calculate spectral clarity improvement
    if 'spectral_contrast_mean' in original_features and 'spectral_contrast_mean' in enhanced_features:
        original_contrast = np.mean(original_features['spectral_contrast_mean'])
        enhanced_contrast = np.mean(enhanced_features['spectral_contrast_mean'])
        contrast_improvement = enhanced_contrast - original_contrast
    else:
        contrast_improvement = 0

    # 2. SNR improvement
    if 'snr_estimate' in original_features and 'snr_estimate' in enhanced_features:
        snr_improvement = enhanced_features['snr_estimate'] - original_features['snr_estimate']
    else:
        snr_improvement = 0

    # 3. Formant clarity improvement
    formant_improvement = 0
    if 'formant_dispersion' in original_features and 'formant_dispersion' in enhanced_features:
        if original_features['formant_dispersion'] > 0:
            formant_improvement = (enhanced_features['formant_dispersion'] /
                                  original_features['formant_dispersion']) - 1

    # 4. Consonant clarity improvement
    consonant_improvement = 0
    if 'high_freq_energy_ratio' in original_features and 'high_freq_energy_ratio' in enhanced_features:
        if original_features['high_freq_energy_ratio'] > 0:
            consonant_improvement = (enhanced_features['high_freq_energy_ratio'] /
                                   original_features['high_freq_energy_ratio']) - 1

    # Calculate overall comprehensibility scores
    try:
        original_score = extract_speech_comprehensibility_score(audio_data, sample_rate)
        enhanced_score = extract_speech_comprehensibility_score(enhanced_audio, sample_rate)
        comprehensibility_improvement = enhanced_score - original_score
    except:
        original_score = 0
        enhanced_score = 0
        comprehensibility_improvement = 0

    # Create enhancement metrics dictionary with detailed information
    metrics = {
        "enhanced": True,
        "confidence": float(confidence),
        "processing_time_ms": float(processing_time),
        "model_type": "user" if user_id else "base",
        "spectral_contrast_improvement": float(contrast_improvement),
        "snr_improvement": float(snr_improvement),
        "formant_improvement": float(formant_improvement),
        "consonant_improvement": float(consonant_improvement),
        "original_comprehensibility": float(original_score),
        "enhanced_comprehensibility": float(enhanced_score),
        "comprehensibility_improvement": float(comprehensibility_improvement)
    }

    # Log comprehensive enhancement metrics
    logger.info(f"Enhancement metrics: confidence={confidence:.2f}, "
               f"comprehensibility: {original_score:.2f} → {enhanced_score:.2f}, "
               f"time={processing_time:.2f}ms")

    return enhanced_audio, metrics


def train_model(unclear_audio_paths, clear_audio_paths, user_id, epochs=50, batch_size=32, job_id=None):
    """
    Train speech enhancement model with extensive validation and metrics.
    Improved with progress tracking, exception handling, and auto-recovery.

    Args:
        unclear_audio_paths (list): Paths to unclear speech audio files
        clear_audio_paths (list): Paths to clear speech audio files
        user_id (str): User ID for personalized model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        job_id (str, optional): Job ID for tracking training progress

    Returns:
        dict: Training metrics
    """
    global _training_jobs, _user_models

    job_id = job_id or f"train_{user_id}_{int(time.time())}"

    # Initialize job tracking
    if job_id:
        _training_jobs[job_id] = {
            "status": "starting",
            "user_id": user_id,
            "start_time": time.time(),
            "progress": 0.0,
            "epochs_completed": 0,
            "total_epochs": epochs
        }

    logger.info(f"Starting training job {job_id} for user {user_id} with {len(unclear_audio_paths)} sample pairs")

    try:
        # Update job status
        if job_id:
            _training_jobs[job_id]["status"] = "preparing_data"
            _training_jobs[job_id]["progress"] = 0.05

        # Get or create user model
        model = get_model(user_id)

        # If existing model is None, create a new one
        if model is None or model.model is None:
            model = SpeechEnhancementModel(user_id=user_id)
            model.build_model()

        # Validate input data
        if len(unclear_audio_paths) != len(clear_audio_paths):
            error_msg = f"Mismatch between unclear ({len(unclear_audio_paths)}) and clear ({len(clear_audio_paths)}) audio files"
            logger.error(error_msg)
            if job_id:
                _training_jobs[job_id]["status"] = "failed"
                _training_jobs[job_id]["error"] = error_msg
            return {"status": "error", "message": error_msg}

        # Log sample details and validate files
        valid_paths = []
        for i, (unclear_path, clear_path) in enumerate(zip(unclear_audio_paths, clear_audio_paths)):
            # Check if files exist
            if not os.path.exists(unclear_path) or not os.path.exists(clear_path):
                logger.warning(f"Training pair {i + 1}: File not found - {unclear_path} or {clear_path}")
                continue

            logger.info(f"Training pair {i + 1}: {Path(unclear_path).name} <-> {Path(clear_path).name}")
            valid_paths.append((unclear_path, clear_path))

        if not valid_paths:
            error_msg = "No valid training pairs found"
            logger.error(error_msg)
            if job_id:
                _training_jobs[job_id]["status"] = "failed"
                _training_jobs[job_id]["error"] = error_msg
            return {"status": "error", "message": error_msg}

        # Unpack valid paths
        valid_unclear_paths = [p[0] for p in valid_paths]
        valid_clear_paths = [p[1] for p in valid_paths]

        # Update job status
        if job_id:
            _training_jobs[job_id]["status"] = "preprocessing"
            _training_jobs[job_id]["progress"] = 0.1
            _training_jobs[job_id]["training_pairs"] = len(valid_paths)

        # Prepare training data with enhanced preprocessing
        logger.info("Preparing training data...")
        try:
            train_data = model.prepare_training_data(valid_unclear_paths, valid_clear_paths)
        except Exception as prep_error:
            logger.error(f"Error in data preparation: {prep_error}")
            # Try again with more conservative preprocessing
            logger.info("Retrying with conservative preprocessing...")
            try:
                # Set a simpler preprocessing pipeline temporarily
                original_preprocess = model.preprocess_audio
                model.preprocess_audio = lambda audio, sr=None: model.preprocess_audio_basic(audio, sr)

                # Try again
                train_data = model.prepare_training_data(valid_unclear_paths, valid_clear_paths)

                # Restore original preprocessing
                model.preprocess_audio = original_preprocess
            except Exception as retry_error:
                error_msg = f"Failed to prepare training data even with conservative preprocessing: {retry_error}"
                logger.error(error_msg)
                if job_id:
                    _training_jobs[job_id]["status"] = "failed"
                    _training_jobs[job_id]["error"] = error_msg
                return {"status": "error", "message": error_msg}

        # Split into train/test with stratified sampling for better validation
        test_size = max(1, int(len(valid_unclear_paths) * 0.2))

        # Update job status
        if job_id:
            _training_jobs[job_id]["status"] = "training"
            _training_jobs[job_id]["progress"] = 0.15

        # Use a better strategy for splitting that ensures representative examples
        # Shuffle with a fixed seed for reproducibility
        np.random.seed(42)
        indices = np.arange(len(train_data[0]))
        np.random.shuffle(indices)

        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        test_data = (
            np.array([train_data[0][i] for i in test_indices]),
            np.array([train_data[1][i] for i in test_indices])
        )
        training_data = (
            np.array([train_data[0][i] for i in train_indices]),
            np.array([train_data[1][i] for i in train_indices])
        )

        logger.info(f"Training data split: {len(train_indices)} training samples, {len(test_indices)} validation samples")

        # Define custom callback for job progress tracking
        class JobProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if job_id:
                    progress = 0.15 + (0.7 * (epoch + 1) / epochs)  # 15% to 85% during training
                    _training_jobs[job_id]["progress"] = min(0.85, progress)
                    _training_jobs[job_id]["epochs_completed"] = epoch + 1
                    _training_jobs[job_id]["current_loss"] = logs.get("loss", 0)
                    _training_jobs[job_id]["current_val_loss"] = logs.get("val_loss", 0)

        # Add learning rate tracker
        lr_tracker = []

        class LearningRateTracker(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Get the current learning rate from the optimizer
                # This works with both newer and older versions of TF
                if hasattr(self.model.optimizer, 'lr'):
                    # Old style
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                elif hasattr(self.model.optimizer, '_learning_rate'):
                    # New style
                    lr = float(tf.keras.backend.get_value(self.model.optimizer._learning_rate))
                elif hasattr(self.model.optimizer, 'learning_rate'):
                    # Another possible approach
                    lr = float(self.model.optimizer.learning_rate)
                else:
                    # Fallback
                    lr = 0.001

                lr_tracker.append(lr)
                logs['lr'] = lr

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            LearningRateTracker(),
            JobProgressCallback()
        ]

        # Train with exception handling
        try:
            history = model.train(training_data, test_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        except Exception as training_error:
            logger.error(f"Error during training: {training_error}")

            # Try to continue with reduced complexity as fallback
            try:
                logger.info("Attempting to recover with simplified model...")
                # Create a new model with simpler architecture
                backup_model = SpeechEnhancementModel(user_id=user_id)
                # Use the existing build_model method but with reduced complexity
                backup_model.build_model(input_shape=training_data[0].shape[1:])

                # Train the simpler model
                history = backup_model.train(training_data, test_data, epochs=max(25, epochs//2),
                                            batch_size=batch_size, callbacks=callbacks)

                # Replace the original model with the backup
                model = backup_model
                logger.info("Successfully recovered with simplified model")
            except Exception as recovery_error:
                # If recovery fails too, report error
                error_msg = f"Training failed and recovery attempt also failed: {recovery_error}"
                logger.error(error_msg)
                if job_id:
                    _training_jobs[job_id]["status"] = "failed"
                    _training_jobs[job_id]["error"] = error_msg
                return {"status": "error", "message": error_msg}

        # Add learning rate to history if it's not there
        if 'lr' not in history:
            history['lr'] = lr_tracker

        # Store training history for visualization
        _training_histories[job_id] = history

        # Update job status
        if job_id:
            _training_jobs[job_id]["status"] = "saving_model"
            _training_jobs[job_id]["progress"] = 0.9

        # Save trained model
        model.save_model(user_specific=True)
        logger.success(f"Model saved for user: {user_id}")

        # Update global dictionary
        with _model_lock:
            _user_models[user_id] = model

        # Create visualization of training progress
        try:
            create_training_visualization(history, user_id)
        except Exception as viz_error:
            logger.warning(f"Error creating training visualization: {viz_error}")

        # Run validation on test samples
        if job_id:
            _training_jobs[job_id]["status"] = "validating"
            _training_jobs[job_id]["progress"] = 0.95

        test_metrics = validate_model_on_samples(model, test_data)

        # Get final metrics with detailed statistics
        metrics = {
            'status': 'success',
            'loss': float(history['loss'][-1]),
            'val_loss': float(history['val_loss'][-1]) if 'val_loss' in history else None,
            'mae': float(history['mean_absolute_error'][-1]),
            'val_mae': float(history['val_mean_absolute_error'][-1]) if 'val_mean_absolute_error' in history else None,
            'clarity': float(history['speech_clarity_metric'][-1]) if 'speech_clarity_metric' in history else None,
            'val_clarity': float(history['val_speech_clarity_metric'][-1]) if 'val_speech_clarity_metric' in history else None,
            'epochs_completed': len(history['loss']),
            'training_samples': len(training_data[0]),
            'validation_samples': len(test_data[0]),
            'initial_loss': float(history['loss'][0]),
            'loss_improvement': float(history['loss'][0] - history['loss'][-1]),
            'training_id': job_id,
            'learning_rate': float(history['lr'][-1]) if 'lr' in history and history['lr'] else 0.001,
            'best_epoch': np.argmin(history['val_loss']) if 'val_loss' in history else np.argmin(history['loss']),
            'model_path': str(model.user_model_path) + '.keras'
        }

        # Update with test metrics
        metrics.update(test_metrics)

        # Update job status
        if job_id:
            _training_jobs[job_id]["status"] = "completed"
            _training_jobs[job_id]["progress"] = 1.0
            _training_jobs[job_id]["metrics"] = metrics
            _training_jobs[job_id]["completed_time"] = time.time()
            _training_jobs[job_id]["duration"] = _training_jobs[job_id]["completed_time"] - _training_jobs[job_id]["start_time"]

        logger.success(
            f"Training completed for user {user_id}: Loss {metrics['loss']:.4f}, Clarity {metrics.get('clarity', 0):.4f}")
        return metrics

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        if job_id:
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["error"] = str(e)
            _training_jobs[job_id]["completed_time"] = time.time()
            _training_jobs[job_id]["duration"] = _training_jobs[job_id]["completed_time"] - _training_jobs[job_id]["start_time"]

        # Return error information
        return {
            "status": "error",
            "message": str(e),
            "job_id": job_id
        }

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

        # Store sample validation with explicit type conversion for JSON serialization
        validation_metrics["sample_validations"].append({
            "sample_index": int(idx),  # Convert from np.int64 to Python int
            "mse": float(mse),         # Convert from np.float to Python float
            "mae": float(mae),
            "clarity": float(clarity)
        })

    # Average metrics with explicit type conversion
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

    def get_training_job_status(job_id):
        """
        Get detailed status information for a training job.

        Args:
            job_id (str): ID of the training job to check

        Returns:
            dict: Job status information or None if job not found
        """
        if job_id in _training_jobs:
            return _training_jobs[job_id]
        return None

    def get_all_training_jobs(user_id=None):
        """
        Get all training jobs, optionally filtered by user ID.

        Args:
            user_id (str, optional): Filter jobs for this user ID

        Returns:
            list: List of training job information dictionaries
        """
        if user_id:
            return [job for job_id, job in _training_jobs.items() if job.get("user_id") == user_id]
        return list(_training_jobs.values())

    def cancel_training_job(job_id):
        """
        Cancel a running training job.

        Args:
            job_id (str): ID of the job to cancel

        Returns:
            bool: True if job was cancelled, False otherwise
        """
        if job_id in _training_jobs and _training_jobs[job_id]["status"] in ["starting", "preparing_data",
                                                                             "preprocessing", "training"]:
            _training_jobs[job_id]["status"] = "cancelled"
            _training_jobs[job_id]["completed_time"] = time.time()
            _training_jobs[job_id]["duration"] = _training_jobs[job_id]["completed_time"] - _training_jobs[job_id][
                "start_time"]
            logger.info(f"Training job {job_id} cancelled")
            return True
        return False

    def clean_up_old_training_jobs(max_age_hours=24):
        """
        Clean up old training jobs to free memory.

        Args:
            max_age_hours (int): Maximum age in hours for jobs to keep

        Returns:
            int: Number of jobs removed
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        jobs_to_remove = []

        for job_id, job in _training_jobs.items():
            # Calculate job age
            start_time = job.get("start_time", 0)
            job_age = current_time - start_time

            # If job is older than max age and is in a terminal state
            if job_age > max_age_seconds and job.get("status") in ["completed", "failed", "cancelled"]:
                jobs_to_remove.append(job_id)

        # Remove old jobs
        for job_id in jobs_to_remove:
            del _training_jobs[job_id]

            # Also remove from training histories if present
            if job_id in _training_histories:
                del _training_histories[job_id]

        logger.info(f"Cleaned up {len(jobs_to_remove)} old training jobs")
        return len(jobs_to_remove)

    def prepare_models_for_shutdown():
        """
        Prepare models for application shutdown.
        Ensures all models are properly saved.
        """
        logger.info("Preparing models for shutdown")

        # Save base model if modified
        if _global_model is not None and _global_model.model is not None:
            try:
                _global_model.save_model(user_specific=False)
                logger.info("Base model saved before shutdown")
            except Exception as e:
                logger.error(f"Error saving base model: {e}")

        # Save all user models
        for user_id, model in _user_models.items():
            try:
                model.save_model(user_specific=True)
                logger.info(f"User model for {user_id} saved before shutdown")
            except Exception as e:
                logger.error(f"Error saving user model for {user_id}: {e}")

        logger.info("Model preparation for shutdown complete")

    def build_model_simple(user_id=None):
        """
        Build a simplified model for fallback purposes.
        Uses a lighter architecture that's more likely to train successfully.

        Args:
            user_id (str, optional): User ID for the model

        Returns:
            SpeechEnhancementModel: The simplified model
        """
        model = SpeechEnhancementModel(user_id=user_id)

        # Create a simpler model architecture
        input_shape = (128, 128, 1)  # Standard input shape for spectrograms

        inputs = tf.keras.layers.Input(shape=input_shape)

        # Simpler encoder
        conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        # Bottleneck
        bottleneck = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)

        # Decoder
        up3 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(bottleneck)
        concat3 = tf.keras.layers.Concatenate()([up3, conv3])
        deconv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat3)

        up2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(deconv3)
        concat2 = tf.keras.layers.Concatenate()([up2, conv2])
        deconv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)

        up1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(deconv2)
        concat1 = tf.keras.layers.Concatenate()([up1, conv1])
        deconv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(concat1)

        # Output layer
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(deconv1)

        # Create and compile model
        tf_model = tf.keras.Model(inputs, outputs)
        tf_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        model.model = tf_model
        return model

    def process_audio_in_batches(audio_data, sample_rate, model, batch_size_ms=500, overlap_ms=100):
        """
        Process long audio in batches to improve memory usage and potentially performance.

        Args:
            audio_data: Audio signal to process
            sample_rate: Sample rate of the audio
            model: SpeechEnhancementModel to use for processing
            batch_size_ms: Size of each batch in milliseconds
            overlap_ms: Overlap between batches in milliseconds

        Returns:
            Enhanced audio with batches combined
        """
        if len(audio_data) < sample_rate:  # If audio is shorter than 1 second
            # Just process directly
            return model.enhance_audio_with_confidence(audio_data, sample_rate)[0]

        # Calculate batch size and overlap in samples
        batch_samples = int(sample_rate * batch_size_ms / 1000)
        overlap_samples = int(sample_rate * overlap_ms / 1000)
        step_samples = batch_samples - overlap_samples

        # Initialize output buffer
        enhanced_audio = np.zeros_like(audio_data)

        # Process in batches with overlap
        for start in range(0, len(audio_data), step_samples):
            end = min(start + batch_samples, len(audio_data))

            # Process this batch
            batch = audio_data[start:end]
            enhanced_batch, _ = model.enhance_audio_with_confidence(batch, sample_rate)

            # Create crossfade window for smooth transitions
            if start > 0 and overlap_samples > 0:
                # Crossfade with previous batch
                crossfade_start = start
                crossfade_end = start + overlap_samples

                # Create crossfade windows
                fade_in = np.linspace(0, 1, overlap_samples)
                fade_out = np.linspace(1, 0, overlap_samples)

                # Apply crossfade
                enhanced_audio[crossfade_start:crossfade_end] = (
                        enhanced_audio[crossfade_start:crossfade_end] * fade_out +
                        enhanced_batch[:overlap_samples] * fade_in
                )

                # Add remaining part of the batch
                if end - (start + overlap_samples) > 0:
                    enhanced_audio[crossfade_end:end] = enhanced_batch[overlap_samples:]
            else:
                # First batch or no overlap
                enhanced_audio[start:end] = enhanced_batch

        return enhanced_audio