"""
Speech Enhancement Model for WillSpeak

This module defines a TensorFlow model for enhancing speech clarity
specifically tailored to users with speech impediments.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate,
    Dropout, BatchNormalization, Reshape, Dense, LSTM, Bidirectional,
    Permute, Activation, Lambda
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import librosa
import librosa.display
import soundfile as sf
from datetime import datetime
import json
import matplotlib.pyplot as plt
from loguru import logger
import sys


class SpeechEnhancementModel:
    """
    A deep learning model for enhancing speech clarity.

    This model is designed to handle speech with various impediments,
    and can be fine-tuned for specific users.
    """

    def __init__(self, user_id=None, model_dir=None):
        """
        Initialize the speech enhancement model.

        Args:
            user_id (str, optional): User ID for personalized model
            model_dir (str, optional): Directory to store models
        """
        self.user_id = user_id or "default"
        self.model_dir = model_dir or os.path.join(os.getcwd(), "models")

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Model paths
        self.base_model_path = os.path.join(self.model_dir, "base_model")
        self.user_model_path = os.path.join(
            self.model_dir, f"user_{self.user_id}_model"
        )

        # Model parameters
        self.sr = 16000  # Sample rate
        self.n_fft = 512  # FFT window size
        self.hop_length = 128  # Hop length
        self.n_mels = 128  # Number of mel bands

        # Initialize model
        self.model = None

        logger.info(f"Initialized SpeechEnhancementModel for user: {self.user_id}")

    def build_model(self, input_shape=(128, 128, 1)):
        """
        Build the enhancement model architecture.

        Args:
            input_shape (tuple): Shape of input spectrogram

        Returns:
            tf.keras.Model: The compiled model
        """
        logger.info(f"Building model with input shape: {input_shape}")

        # Input layer
        inputs = Input(shape=input_shape)

        # Encoder path
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # Bottleneck
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

        # Decoder path with skip connections
        up5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
        up5 = Concatenate()([up5, conv3])
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

        up6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5)
        up6 = Concatenate()([up6, conv2])
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

        up7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
        up7 = Concatenate()([up7, conv1])
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

        # Output layer
        outputs = Conv2D(1, (1, 1), activation='tanh')(conv7)

        # Create and compile model
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        self.model = model
        logger.info(f"Model built: {model.summary()}")
        return model

    def load_model(self, user_specific=True):
        """
        Load a pre-trained model.

        Args:
            user_specific (bool): Whether to load user-specific model

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        model_path = self.user_model_path if user_specific else self.base_model_path

        if not os.path.exists(model_path):
            if user_specific and os.path.exists(self.base_model_path):
                logger.info(f"User model not found. Loading base model instead.")
                model_path = self.base_model_path
            else:
                logger.warning(f"No pre-trained model found at {model_path}")
                return False

        try:
            self.model = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def save_model(self, user_specific=True):
        """
        Save the current model.

        Args:
            user_specific (bool): Whether to save as user-specific model

        Returns:
            bool: True if saved successfully, False otherwise
        """
        if self.model is None:
            logger.error("No model to save")
            return False

        model_path = self.user_model_path if user_specific else self.base_model_path

        try:
            self.model.save(model_path)
            logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def preprocess_audio(self, audio_data, sr=None):
        """
        Preprocess audio for model input.

        Args:
            audio_data (np.array): Audio waveform
            sr (int, optional): Sample rate of audio

        Returns:
            np.array: Mel spectrogram suitable for model input
        """
        sr = sr or self.sr

        # Resample if needed
        if sr != self.sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sr)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [-1, 1] range
        normalized_spec = 2 * ((log_mel_spec - log_mel_spec.min()) /
                               (log_mel_spec.max() - log_mel_spec.min())) - 1

        # Ensure proper shape
        normalized_spec = np.expand_dims(normalized_spec, axis=-1)

        return normalized_spec

    def postprocess_spectrogram(self, enhanced_spec, original_phase=None):
        """
        Convert enhanced spectrogram back to audio.

        Args:
            enhanced_spec (np.array): Enhanced mel spectrogram
            original_phase (np.array, optional): Phase information from original audio

        Returns:
            np.array: Enhanced audio waveform
        """
        # Remove extra dimension
        enhanced_spec = np.squeeze(enhanced_spec)

        # Convert from [-1, 1] to dB scale
        enhanced_db = ((enhanced_spec + 1) / 2) * 80 - 80

        # Convert from dB to power
        enhanced_mel = librosa.db_to_power(enhanced_db)

        # Convert mel spectrogram to STFT magnitude
        enhanced_mag = librosa.feature.inverse.mel_to_stft(
            enhanced_mel,
            sr=self.sr,
            n_fft=self.n_fft
        )

        # Griffin-Lim algorithm to estimate phase if not provided
        if original_phase is None:
            enhanced_audio = librosa.griffinlim(
                enhanced_mag,
                hop_length=self.hop_length,
                n_iter=32
            )
        else:
            # Use original phase information
            enhanced_stft = enhanced_mag * np.exp(1j * original_phase)
            enhanced_audio = librosa.istft(
                enhanced_stft,
                hop_length=self.hop_length
            )

        return enhanced_audio

    def enhance_audio(self, audio_data, sr=None):
        """
        Enhance audio using the trained model.

        Args:
            audio_data (np.array): Audio waveform to enhance
            sr (int, optional): Sample rate of audio

        Returns:
            np.array: Enhanced audio waveform
        """
        if self.model is None:
            success = self.load_model()
            if not success:
                logger.error("No model available for enhancement")
                return audio_data

        sr = sr or self.sr

        # Save original audio length for later
        original_length = len(audio_data)

        # Get original phase information
        stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        original_phase = np.angle(stft)

        # Preprocess audio into mel spectrogram
        mel_spec = self.preprocess_audio(audio_data, sr)

        # Pad or crop to expected size
        target_height, target_width = 128, 128
        if mel_spec.shape[0] < target_height or mel_spec.shape[1] < target_width:
            # Pad if too small
            padded_spec = np.zeros((target_height, target_width, 1))
            h, w = min(mel_spec.shape[0], target_height), min(mel_spec.shape[1], target_width)
            padded_spec[:h, :w, :] = mel_spec[:h, :w, :]
            mel_spec = padded_spec
        elif mel_spec.shape[0] > target_height or mel_spec.shape[1] > target_width:
            # Crop if too large
            mel_spec = mel_spec[:target_height, :target_width, :]

        # Reshape for model input
        model_input = np.expand_dims(mel_spec, axis=0)

        # Get model prediction
        enhanced_spec = self.model.predict(model_input)[0]

        # Convert back to audio
        enhanced_audio = self.postprocess_spectrogram(enhanced_spec, original_phase)

        # Ensure output length matches original
        if len(enhanced_audio) < original_length:
            enhanced_audio = np.pad(enhanced_audio, (0, original_length - len(enhanced_audio)))
        else:
            enhanced_audio = enhanced_audio[:original_length]

        return enhanced_audio

    def train(self, train_data, test_data=None, epochs=50, batch_size=32):
        """
        Train the enhancement model.

        Args:
            train_data (tuple): Tuple of (noisy_specs, clean_specs) for training
            test_data (tuple, optional): Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training

        Returns:
            dict: Training history
        """
        if self.model is None:
            input_shape = train_data[0].shape[1:]
            self.build_model(input_shape)

        # Create callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'checkpoint.h5'),
                save_best_only=True
            ),
            TensorBoard(
                log_dir=os.path.join(self.model_dir, 'logs',
                                     datetime.now().strftime('%Y%m%d-%H%M%S'))
            )
        ]

        # Train model
        logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}")

        history = self.model.fit(
            train_data[0], train_data[1],
            validation_data=test_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        # Save trained model
        self.save_model(user_specific=True)

        # Save training metrics
        metrics = {
            'loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1]) if test_data is not None else None,
            'mae': float(history.history['mean_absolute_error'][-1]),
            'val_mae': float(history.history['val_mean_absolute_error'][-1]) if test_data is not None else None,
            'epochs': len(history.history['loss']),
            'timestamp': datetime.now().isoformat()
        }

        # Save metrics to file
        metrics_path = os.path.join(self.model_dir, f"user_{self.user_id}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training completed. Loss: {metrics['loss']:.4f}, MAE: {metrics['mae']:.4f}")
        return history.history

    def prepare_training_data(self, unclear_audio_paths, clear_audio_paths):
        """
        Prepare training data from audio file pairs.

        Args:
            unclear_audio_paths (list): Paths to unclear/impaired speech files
            clear_audio_paths (list): Paths to corresponding clear speech files

        Returns:
            tuple: Training data as (X_train, y_train)
        """
        if len(unclear_audio_paths) != len(clear_audio_paths):
            raise ValueError("Number of unclear and clear audio files must match")

        unclear_specs = []
        clear_specs = []

        # Process each audio pair
        for i, (unclear_path, clear_path) in enumerate(zip(unclear_audio_paths, clear_audio_paths)):
            logger.info(f"Processing training pair {i + 1}/{len(unclear_audio_paths)}")

            # Load audio files
            unclear_audio, sr = librosa.load(unclear_path, sr=self.sr)
            clear_audio, _ = librosa.load(clear_path, sr=self.sr)

            # Make sure both audios have the same length
            min_length = min(len(unclear_audio), len(clear_audio))
            unclear_audio = unclear_audio[:min_length]
            clear_audio = clear_audio[:min_length]

            # Convert to mel spectrograms
            unclear_spec = self.preprocess_audio(unclear_audio)
            clear_spec = self.preprocess_audio(clear_audio)

            # Ensure consistent shape
            target_shape = (128, 128, 1)
            if unclear_spec.shape[:2] != target_shape[:2] or clear_spec.shape[:2] != target_shape[:2]:
                # Pad or crop to target shape
                temp_unclear = np.zeros(target_shape)
                temp_clear = np.zeros(target_shape)

                h, w = min(unclear_spec.shape[0], target_shape[0]), min(unclear_spec.shape[1], target_shape[1])
                temp_unclear[:h, :w, :] = unclear_spec[:h, :w, :]
                temp_clear[:h, :w, :] = clear_spec[:h, :w, :]

                unclear_spec = temp_unclear
                clear_spec = temp_clear

            unclear_specs.append(unclear_spec)
            clear_specs.append(clear_spec)

        # Convert to numpy arrays
        X_train = np.array(unclear_specs)
        y_train = np.array(clear_specs)

        logger.info(f"Prepared training data: {X_train.shape}, {y_train.shape}")
        return X_train, y_train

    def visualize_enhancement(self, audio_path, output_path=None):
        """
        Visualize the enhancement results.

        Args:
            audio_path (str): Path to audio file to enhance
            output_path (str, optional): Path to save enhanced audio

        Returns:
            tuple: Original and enhanced audio
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr)

        # Enhance audio
        enhanced_audio = self.enhance_audio(audio, sr)

        # Save enhanced audio if requested
        if output_path:
            sf.write(output_path, enhanced_audio, self.sr)
            logger.info(f"Enhanced audio saved to {output_path}")

        # Create spectrograms
        orig_mel = librosa.feature.melspectrogram(y=audio, sr=self.sr)
        enh_mel = librosa.feature.melspectrogram(y=enhanced_audio, sr=self.sr)

        orig_db = librosa.power_to_db(orig_mel, ref=np.max)
        enh_db = librosa.power_to_db(enh_mel, ref=np.max)

        # Plot spectrograms
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        librosa.display.specshow(orig_db, sr=self.sr, x_axis='time', y_axis='mel')
        plt.title('Original Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2, 1, 2)
        librosa.display.specshow(enh_db, sr=self.sr, x_axis='time', y_axis='mel')
        plt.title('Enhanced Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path + '.png')

        plt.close()

        return audio, enhanced_audio