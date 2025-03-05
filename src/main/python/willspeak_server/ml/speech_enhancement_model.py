"""
Enhanced Speech Enhancement Model for WillSpeak

This module provides an improved TensorFlow model for enhancing speech clarity
specifically tailored to users with speech impediments.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate,
    Dropout, BatchNormalization, Reshape, Dense, LSTM, Bidirectional,
    Permute, Activation, Lambda, Add, Multiply, AveragePooling2D
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import librosa
import librosa.display
import soundfile as sf
from datetime import datetime
import json
import matplotlib.pyplot as plt
from loguru import logger
import sys
from pathlib import Path


class SpeechEnhancementModel:
    """
    An advanced deep learning model for enhancing speech clarity.

    This model is specifically designed to handle speech with various impediments,
    including slurred speech and limited muscle control issues.
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

        # Initialize model and tokenizer
        self.model = None

        logger.info(f"Initialized SpeechEnhancementModel for user: {self.user_id}")

        # Track model confidence and performance
        self.last_confidence_score = 0.0
        self.enhancement_metrics = {
            "total_processed": 0,
            "avg_confidence": 0.0,
            "high_confidence_count": 0,  # Confidence > 0.7
            "processing_times_ms": []
        }

    def build_model(self, input_shape=(128, 128, 1)):
        """
        Build the enhanced speech model architecture with attention mechanisms.

        Args:
            input_shape (tuple): Shape of input spectrogram

        Returns:
            tf.keras.Model: The compiled model
        """
        logger.info(f"Building improved model with input shape: {input_shape}")

        # Input layer
        inputs = Input(shape=input_shape, name="input_spectrogram")

        # -- ENCODER PATH with Attention mechanisms --

        # First level
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # Second level with attention
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        # Add self-attention mechanism
        attention2 = self.self_attention_block(conv2, 64)
        pool2 = MaxPooling2D(pool_size=(2, 2))(attention2)

        # Third level with increased filters
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        attention3 = self.self_attention_block(conv3, 128)
        pool3 = MaxPooling2D(pool_size=(2, 2))(attention3)

        # Fourth level (deeper than original)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        attention4 = self.self_attention_block(conv4, 256)
        pool4 = MaxPooling2D(pool_size=(2, 2))(attention4)

        # -- BOTTLENECK with dilated convolutions --
        # Dilated convolutions expand the receptive field to capture wider context
        bottleneck = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same')(pool4)
        bottleneck = BatchNormalization()(bottleneck)
        bottleneck = Dropout(0.3)(bottleneck)
        bottleneck = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same')(bottleneck)
        bottleneck = BatchNormalization()(bottleneck)

        # -- DECODER PATH with skip connections --

        # Level 4 up
        up4 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bottleneck)
        merge4 = Concatenate()([up4, attention4])
        deconv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge4)
        deconv4 = BatchNormalization()(deconv4)
        deconv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(deconv4)
        deconv4 = BatchNormalization()(deconv4)

        # Level 3 up
        up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(deconv4)
        merge3 = Concatenate()([up3, attention3])
        deconv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge3)
        deconv3 = BatchNormalization()(deconv3)
        deconv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(deconv3)
        deconv3 = BatchNormalization()(deconv3)

        # Level 2 up
        up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(deconv3)
        merge2 = Concatenate()([up2, attention2])
        deconv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge2)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(deconv2)
        deconv2 = BatchNormalization()(deconv2)

        # Level 1 up - return to original size
        up1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(deconv2)
        merge1 = Concatenate()([up1, conv1])
        deconv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge1)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(deconv1)
        deconv1 = BatchNormalization()(deconv1)

        # Output layer - using sigmoid activation for better quality at spectral level
        outputs = Conv2D(1, (1, 1), activation='sigmoid', name="enhanced_spectrogram")(deconv1)

        # Create model
        model = Model(inputs, outputs)

        # Define custom loss function that focuses on speech clarity
        def perceptual_loss(y_true, y_pred):
            # Basic MSE loss
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

            # Spectral loss component to focus on frequency characteristics
            # This helps maintain phonetic clarity
            y_true_flat = tf.reshape(y_true, [-1, input_shape[0], input_shape[1]])
            y_pred_flat = tf.reshape(y_pred, [-1, input_shape[0], input_shape[1]])

            # Focus more on mid-frequencies (important for speech)
            # Create a weighting mask that emphasizes mid-frequencies
            freq_importance = np.ones(input_shape[0])
            # 20-60% of frequency range gets higher weight (consonants, vowel transitions)
            mid_freq_start = int(input_shape[0] * 0.2)
            mid_freq_end = int(input_shape[0] * 0.6)
            freq_importance[mid_freq_start:mid_freq_end] = 2.0
            freq_importance = tf.constant(freq_importance, dtype=tf.float32)
            freq_importance = tf.reshape(freq_importance, [input_shape[0], 1])

            # Apply frequency weighting
            weighted_true = y_true_flat * freq_importance
            weighted_pred = y_pred_flat * freq_importance

            # Spectral loss with frequency weighting
            spectral_loss = tf.reduce_mean(tf.abs(weighted_true - weighted_pred))

            # Combined loss with higher weight on spectral component
            return mse_loss + 1.5 * spectral_loss

        # Custom metric to track speech clarity improvements
        def speech_clarity_metric(y_true, y_pred):
            # Calculate correlation between true and predicted spectrograms
            # Higher correlation indicates better clarity
            y_true_flat = tf.reshape(y_true, [-1])
            y_pred_flat = tf.reshape(y_pred, [-1])

            # Calculate Pearson correlation
            true_mean = tf.reduce_mean(y_true_flat)
            pred_mean = tf.reduce_mean(y_pred_flat)

            true_centered = y_true_flat - true_mean
            pred_centered = y_pred_flat - pred_mean

            numerator = tf.reduce_sum(true_centered * pred_centered)
            denominator = tf.sqrt(tf.reduce_sum(tf.square(true_centered)) *
                                  tf.reduce_sum(tf.square(pred_centered)))

            # Avoid division by zero
            denominator = tf.maximum(denominator, 1e-12)
            correlation = numerator / denominator

            return correlation

        # Compile with Adam optimizer and perceptual loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=perceptual_loss,
            metrics=['mean_absolute_error', speech_clarity_metric]
        )

        self.model = model
        model.summary(print_fn=logger.info)
        return model

    def self_attention_block(self, x, filters):
        """
        Create a self-attention block to focus on important features using Keras operations.

        Args:
            x: Input tensor
            filters: Number of filters

        Returns:
            Tensor with attention applied
        """
        # Channel attention
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
        avg_pool = tf.keras.layers.Reshape((1, 1, filters))(avg_pool)
        avg_pool = tf.keras.layers.Conv2D(filters // 8, kernel_size=1, activation='relu')(avg_pool)
        avg_pool = tf.keras.layers.Conv2D(filters, kernel_size=1, activation='sigmoid')(avg_pool)

        # Apply channel attention
        channel_attention = tf.keras.layers.Multiply()([x, avg_pool])

        # Spatial attention
        # Use Keras Lambda layer to wrap TensorFlow operations
        avg_pool_spatial = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
        )(x)
        max_pool_spatial = tf.keras.layers.Lambda(
            lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
        )(x)

        concat = tf.keras.layers.Concatenate()([avg_pool_spatial, max_pool_spatial])
        spatial_attention = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)

        # Apply spatial attention
        spatial_attention_output = tf.keras.layers.Multiply()([channel_attention, spatial_attention])

        # Add residual connection
        return tf.keras.layers.Add()([x, spatial_attention_output])

    def load_model(self, user_specific=True):
        """
        Load a pre-trained model with proper extension handling.

        Args:
            user_specific (bool): Whether to load user-specific model

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        model_base_path = self.user_model_path if user_specific else self.base_model_path

        # Try different possible extensions
        possible_paths = [
            model_base_path,
            model_base_path + '.keras',
            model_base_path + '.h5'
        ]

        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                    self.model = load_model(model_path, compile=False)

                    # Recompile the model with custom loss and metrics
                    def perceptual_loss(y_true, y_pred):
                        # Basic MSE loss
                        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

                        # Reshape inputs for spectral loss
                        y_true_flat = tf.reshape(y_true, [-1, 128, 128])
                        y_pred_flat = tf.reshape(y_pred, [-1, 128, 128])

                        # Create frequency weighting
                        freq_importance = np.ones(128)
                        mid_freq_start = int(128 * 0.2)
                        mid_freq_end = int(128 * 0.6)
                        freq_importance[mid_freq_start:mid_freq_end] = 2.0
                        freq_importance = tf.constant(freq_importance, dtype=tf.float32)
                        freq_importance = tf.reshape(freq_importance, [128, 1])

                        # Apply frequency weighting
                        weighted_true = y_true_flat * freq_importance
                        weighted_pred = y_pred_flat * freq_importance

                        # Spectral loss with frequency weighting
                        spectral_loss = tf.reduce_mean(tf.abs(weighted_true - weighted_pred))

                        # Combined loss
                        return mse_loss + 1.5 * spectral_loss

                    def speech_clarity_metric(y_true, y_pred):
                        # Calculate correlation between true and predicted spectrograms
                        y_true_flat = tf.reshape(y_true, [-1])
                        y_pred_flat = tf.reshape(y_pred, [-1])

                        # Calculate Pearson correlation
                        true_mean = tf.reduce_mean(y_true_flat)
                        pred_mean = tf.reduce_mean(y_pred_flat)

                        true_centered = y_true_flat - true_mean
                        pred_centered = y_pred_flat - pred_mean

                        numerator = tf.reduce_sum(true_centered * pred_centered)
                        denominator = tf.sqrt(tf.reduce_sum(tf.square(true_centered)) *
                                              tf.reduce_sum(tf.square(pred_centered)))

                        # Avoid division by zero
                        denominator = tf.maximum(denominator, 1e-12)
                        correlation = numerator / denominator

                        return correlation

                    # Recompile the model
                    self.model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss=perceptual_loss,
                        metrics=['mean_absolute_error', speech_clarity_metric]
                    )

                    logger.info(f"Loaded model from {model_path}")
                    return True
                except Exception as e:
                    logger.error(f"Error loading model from {model_path}: {str(e)}")
                    # Continue trying other paths

        # If we got here, no model was found or loaded successfully
        if user_specific and os.path.exists(self.base_model_path + '.keras'):
            logger.info(f"User model not found. Loading base model instead.")
            return self.load_model(user_specific=False)

        logger.warning(f"No pre-trained model found")
        return False

    def apply_emphasis_filter(self, audio_data, pre_emphasis=0.97):
        """
        Apply pre-emphasis filter to enhance higher frequencies.
        This improves clarity of consonants.

        Args:
            audio_data: Input audio signal
            pre_emphasis: Pre-emphasis coefficient (typically 0.95-0.97)

        Returns:
            Filtered audio signal
        """
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        return emphasized_audio

    def denoise_speech(self, audio_data, noise_reduction_factor=0.5):
        """
        Simple spectral subtraction-based noise reduction.

        Args:
            audio_data: Input audio signal
            noise_reduction_factor: Reduction factor (0-1)

        Returns:
            Denoised audio signal
        """
        # Calculate short-time Fourier transform
        stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)

        # Get magnitude and phase
        magnitude, phase = librosa.magphase(stft)

        # Estimate noise from the beginning of the signal (assuming first 0.5s is noise)
        noise_frames = int(0.5 * self.sr / self.hop_length)
        if noise_frames < magnitude.shape[1]:
            noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        else:
            # If signal is too short, use a small constant
            noise_estimate = np.mean(magnitude) * 0.1 * np.ones((magnitude.shape[0], 1))

        # Apply spectral subtraction with flooring
        magnitude_reduced = np.maximum(magnitude - noise_reduction_factor * noise_estimate, 0.01 * magnitude)

        # Reconstruct signal
        stft_reduced = magnitude_reduced * phase
        denoised_audio = librosa.istft(stft_reduced, hop_length=self.hop_length)

        # Ensure the output length matches the input length
        if len(denoised_audio) > len(audio_data):
            denoised_audio = denoised_audio[:len(audio_data)]
        elif len(denoised_audio) < len(audio_data):
            denoised_audio = np.pad(denoised_audio, (0, len(audio_data) - len(denoised_audio)))

        return denoised_audio

    def postprocess_spectrogram(self, enhanced_spec, original_phase):
        """
        Convert the enhanced spectrogram back to time domain using original phase information.

        Args:
            enhanced_spec: Enhanced mel spectrogram (output from the model)
            original_phase: Original phase information from the STFT

        Returns:
            Time-domain audio signal
        """
        logger.info("Postprocessing enhanced spectrogram to audio")

        # If the enhanced_spec is a 3D tensor with batch dimension, remove it
        if len(enhanced_spec.shape) == 3:
            enhanced_spec = enhanced_spec[0]  # Remove batch dimension

        # If the enhanced_spec has a channel dimension, remove it
        if len(enhanced_spec.shape) == 3:
            enhanced_spec = enhanced_spec[:, :, 0]

        # Convert from normalized range [0, 1] back to log-mel scale
        log_mel_spec = enhanced_spec * 80 - 80  # Assuming 80dB dynamic range

        # Convert log-mel spectrogram back to magnitude spectrogram (approximate inverse)
        # Note: This is an approximation since mel to linear conversion is lossy
        mel_basis = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels)
        magnitude_spec = np.dot(mel_basis.T, librosa.db_to_power(log_mel_spec))

        # Combine with original phase to reconstruct complex STFT
        complex_stft = magnitude_spec * np.exp(1j * original_phase)

        # Convert back to time domain
        reconstructed_audio = librosa.istft(
            complex_stft,
            hop_length=self.hop_length,
            win_length=self.n_fft
        )

        # Normalize the output
        reconstructed_audio = librosa.util.normalize(reconstructed_audio)

        logger.info(f"Postprocessing complete: Generated {len(reconstructed_audio)} samples")
        return reconstructed_audio
    def compress_dynamic_range(self, audio_data, threshold=-20, ratio=2, attack=0.01, release=0.1):
        """
        Apply dynamic range compression to make quieter sounds more audible.

        Args:
            audio_data: Input audio signal
            threshold: Threshold in dB
            ratio: Compression ratio
            attack: Attack time in seconds
            release: Release time in seconds

        Returns:
            Compressed audio signal
        """
        # Convert threshold to linear amplitude
        threshold_linear = 10 ** (threshold / 20.0)

        # Calculate envelope
        abs_audio = np.abs(audio_data)

        # Apply simple envelope follower
        envelope = np.zeros_like(audio_data)
        attack_coef = np.exp(-1.0 / (self.sr * attack))
        release_coef = np.exp(-1.0 / (self.sr * release))

        for i in range(1, len(audio_data)):
            if abs_audio[i] > envelope[i - 1]:
                # Attack phase
                envelope[i] = attack_coef * envelope[i - 1] + (1 - attack_coef) * abs_audio[i]
            else:
                # Release phase
                envelope[i] = release_coef * envelope[i - 1] + (1 - release_coef) * abs_audio[i]

        # Compress when envelope exceeds threshold
        gain = np.ones_like(audio_data)
        mask = envelope > threshold_linear
        gain[mask] = (envelope[mask] / threshold_linear) ** (1 / ratio - 1)

        # Apply gain
        compressed_audio = audio_data * gain

        # Normalize to avoid clipping
        if np.max(np.abs(compressed_audio)) > 1.0:
            compressed_audio = compressed_audio / np.max(np.abs(compressed_audio))

        return compressed_audio

    def enhance_formants(self, audio_data):
        """
        Enhance formants (resonant frequencies) to improve vowel clarity.

        Args:
            audio_data: Input audio signal

        Returns:
            Formant-enhanced audio signal
        """
        # Calculate STFT
        stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude, phase = librosa.magphase(stft)

        # Frequency bands that typically contain formants
        # First formant (F1): ~500Hz, Second formant (F2): ~1500Hz
        f1_range = (int(500 * self.n_fft / self.sr), int(800 * self.n_fft / self.sr))
        f2_range = (int(1500 * self.n_fft / self.sr), int(2500 * self.n_fft / self.sr))

        # Boost formant ranges
        magnitude[f1_range[0]:f1_range[1], :] *= 1.2  # 20% boost for F1
        magnitude[f2_range[0]:f2_range[1], :] *= 1.3  # 30% boost for F2

        # Reconstruct signal
        enhanced_stft = magnitude * phase
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)

        # Match the original length
        if len(enhanced_audio) > len(audio_data):
            enhanced_audio = enhanced_audio[:len(audio_data)]
        elif len(enhanced_audio) < len(audio_data):
            enhanced_audio = np.pad(enhanced_audio, (0, len(audio_data) - len(enhanced_audio)))

        return enhanced_audio

    def enhance_audio_with_confidence(self, audio_data, sr=None):
        """
        Enhance audio with confidence metrics.

        Args:
            audio_data (np.array): Audio waveform to enhance
            sr (int, optional): Sample rate of audio

        Returns:
            tuple: (enhanced_audio, confidence_score)
        """
        if self.model is None:
            success = self.load_model()
            if not success:
                logger.error("No model available for enhancement")
                return audio_data, 0.0

        sr = sr or self.sr

        # Save original audio length for later
        original_length = len(audio_data)

        # Get original phase information
        stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        original_phase = np.angle(stft)
        original_magnitude = np.abs(stft)

        # Enhanced preprocessing for the input audio
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

        # Calculate confidence score
        # This measures how effectively the model is changing the input
        # Higher value = more confident enhancement
        input_output_diff = np.mean(np.abs(enhanced_spec - mel_spec))
        clarity_improvement = self.calculate_clarity_improvement(mel_spec, enhanced_spec)

        # Normalize to 0-1 range
        # Higher difference means more actual enhancement (up to a point)
        # We want a reasonable amount of change, but not too drastic
        diff_confidence = 1.0 - np.exp(-5.0 * input_output_diff)  # Exponential scaling

        # Combine metrics for final confidence
        confidence = 0.7 * clarity_improvement + 0.3 * diff_confidence

        # Update metrics
        self.last_confidence_score = confidence
        self.enhancement_metrics["total_processed"] += 1
        self.enhancement_metrics["avg_confidence"] = ((self.enhancement_metrics["avg_confidence"] *
                                                       (self.enhancement_metrics["total_processed"] - 1)) +
                                                      confidence) / self.enhancement_metrics["total_processed"]
        if confidence > 0.7:
            self.enhancement_metrics["high_confidence_count"] += 1

        # Enhanced postprocessing with original phase
        try:
            enhanced_audio = self.postprocess_spectrogram(enhanced_spec, original_phase)
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            # Fall back to original audio if postprocessing fails
            enhanced_audio = audio_data

        # Ensure output length matches original
        if len(enhanced_audio) < original_length:
            enhanced_audio = np.pad(enhanced_audio, (0, original_length - len(enhanced_audio)))
        else:
            enhanced_audio = enhanced_audio[:original_length]

        return enhanced_audio, confidence

    def calculate_clarity_improvement(self, input_spec, enhanced_spec):
        """
        Calculate how much the clarity has been improved.

        Args:
            input_spec: Input mel spectrogram
            enhanced_spec: Enhanced mel spectrogram

        Returns:
            float: Clarity improvement score (0-1)
        """
        # Calculate spectral contrast before and after
        # Higher spectral contrast usually means clearer speech
        input_std = np.std(input_spec)
        enhanced_std = np.std(enhanced_spec)

        # Calculate improvement ratio (avoid division by zero)
        if input_std < 1e-6:
            return 0.5  # Neutral confidence if input has no variation

        std_improvement = enhanced_std / input_std

        # Normalize to 0-1 scale with logistic function
        # Values around 1.0-1.5 are considered good improvements
        # Too high might indicate distortion
        normalized_improvement = 1.0 / (1.0 + np.exp(-2.0 * (std_improvement - 1.0)))

        # Also check clarity in important speech frequency ranges (formants)
        formant_clarity = self.measure_formant_clarity(enhanced_spec)

        # Combine metrics
        clarity_score = 0.6 * normalized_improvement + 0.4 * formant_clarity

        return min(1.0, max(0.0, clarity_score))  # Ensure in 0-1 range

    def measure_formant_clarity(self, spec):
        """
        Measures the clarity of speech formants in the spectrogram.

        Args:
            spec: Mel spectrogram

        Returns:
            float: Formant clarity score (0-1)
        """
        # Focus on frequency bands important for speech intelligibility
        # First formant: ~500-800 Hz
        # Second formant: ~1000-2000 Hz
        f1_indices = slice(int(128 * 0.1), int(128 * 0.2))
        f2_indices = slice(int(128 * 0.2), int(128 * 0.4))

        # Calculate contrast in these regions
        f1_contrast = np.std(spec[f1_indices, :])
        f2_contrast = np.std(spec[f2_indices, :])

        # Higher contrast typically means clearer formants
        formant_clarity = 0.5 * (f1_contrast + f2_contrast)

        # Normalize to 0-1 range
        normalized_clarity = min(1.0, formant_clarity * 5.0)

        return normalized_clarity

    def train(self, train_data, test_data=None, epochs=50, batch_size=32, callbacks=None):
        """
        Train the enhancement model with robust handling for small datasets.

        Args:
            train_data (tuple): Tuple of (noisy_specs, clean_specs) for training
            test_data (tuple, optional): Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks (list): Additional callbacks for training

        Returns:
            dict: Training history
        """
        if self.model is None:
            input_shape = train_data[0].shape[1:]
            self.build_model(input_shape)

        # Adjust batch size if dataset is smaller than batch_size
        num_samples = train_data[0].shape[0]
        if num_samples < batch_size:
            batch_size = max(1, num_samples)
            logger.info(f"Adjusted batch_size to {batch_size} due to small dataset ({num_samples} samples)")

        # Define callbacks
        default_callbacks = [
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.keras'),
                save_best_only=True,
                monitor='val_loss' if test_data is not None else 'loss',
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss' if test_data is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if test_data is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(self.model_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S')),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]

        # Combine with any user-provided callbacks
        if callbacks:
            training_callbacks = default_callbacks + callbacks
        else:
            training_callbacks = default_callbacks

        # Training call with progress logging
        logger.info(f"Starting training with {num_samples} samples, batch size {batch_size}, {epochs} epochs")
        history = self.model.fit(
            train_data[0], train_data[1],
            validation_data=test_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=training_callbacks,
            verbose=1
        ).history

        # Add learning rate to history if it's not there
        if 'lr' not in history:
            history['lr'] = [float(tf.keras.backend.get_value(self.model.optimizer.lr)) for _ in
                             range(len(history['loss']))]

        logger.info(f"Training completed after {len(history['loss'])} epochs")

        # Create training metrics summary
        metrics_summary = {
            'final_loss': float(history['loss'][-1]),
            'final_mae': float(history['mean_absolute_error'][-1]),
            'loss_improvement': float(history['loss'][0] - history['loss'][-1]),
            'epochs_completed': len(history['loss']),
        }

        if test_data is not None:
            metrics_summary.update({
                'final_val_loss': float(history['val_loss'][-1]),
                'final_val_mae': float(history['val_mean_absolute_error'][-1]),
                'val_loss_improvement': float(history['val_loss'][0] - history['val_loss'][-1]),
            })

        logger.info(f"Training metrics: {metrics_summary}")
        return history

    def prepare_training_data(self, unclear_audio_paths, clear_audio_paths):
        """
        Prepare training data from audio file pairs with enhanced preprocessing and validation.

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

        # Log total files to process
        total_files = len(unclear_audio_paths)
        logger.info(f"Preparing training data from {total_files} file pairs")

        valid_pairs = 0

        for i, (unclear_path, clear_path) in enumerate(zip(unclear_audio_paths, clear_audio_paths)):
            logger.debug(f"Processing file pair {i + 1}/{total_files}: {os.path.basename(unclear_path)}")
            try:
                # Load audio files
                unclear_audio, sr = librosa.load(unclear_path, sr=self.sr)
                clear_audio, _ = librosa.load(clear_path, sr=self.sr)

                # Make sure both audios have the same length
                min_length = min(len(unclear_audio), len(clear_audio))
                unclear_audio = unclear_audio[:min_length]
                clear_audio = clear_audio[:min_length]

                # Skip very short audio samples
                if min_length < 0.5 * self.sr:  # Less than 0.5 seconds
                    logger.warning(f"Skipping pair {i} - audio too short: {min_length / self.sr:.2f}s")
                    continue

                # Apply enhanced preprocessing
                unclear_spec = self.preprocess_audio(unclear_audio)

                # Minimal preprocessing for clear audio (target)
                clear_mel_spec = librosa.feature.melspectrogram(
                    y=clear_audio,
                    sr=self.sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
                clear_log_mel = librosa.power_to_db(clear_mel_spec, ref=np.max)
                clear_spec = (clear_log_mel - clear_log_mel.min()) / (clear_log_mel.max() - clear_log_mel.min() + 1e-8)
                clear_spec = np.expand_dims(clear_spec, axis=-1)

                # Validate spectrograms for variance
                if np.max(unclear_spec) - np.min(unclear_spec) < 1e-6:
                    logger.warning(f"Unclear spectrogram for pair {i} has very low variance, skipping")
                    continue
                if np.max(clear_spec) - np.min(clear_spec) < 1e-6:
                    logger.warning(f"Clear spectrogram for pair {i} has very low variance, skipping")
                    continue

                # Ensure consistent shape
                target_shape = (128, 128, 1)
                if unclear_spec.shape[:2] != target_shape[:2] or clear_spec.shape[:2] != target_shape[:2]:
                    temp_unclear = np.zeros(target_shape)
                    temp_clear = np.zeros(target_shape)
                    h, w = min(unclear_spec.shape[0], target_shape[0]), min(unclear_spec.shape[1], target_shape[1])
                    temp_unclear[:h, :w, :] = unclear_spec[:h, :w, :]
                    temp_clear[:h, :w, :] = clear_spec[:h, :w, :]

                    unclear_spec = temp_unclear
                    clear_spec = temp_clear

                unclear_specs.append(unclear_spec)
                clear_specs.append(clear_spec)
                valid_pairs += 1

            except Exception as e:
                logger.error(f"Error processing audio pair {i + 1}: {str(e)}")
                continue

        if len(unclear_specs) == 0:
            raise ValueError("No valid training pairs found after preprocessing")

        X_train = np.array(unclear_specs)
        y_train = np.array(clear_specs)

        logger.info(f"Prepared {valid_pairs} valid training pairs out of {total_files} total pairs")
        logger.info(f"Training data shapes: X={X_train.shape}, Y={y_train.shape}")

        return X_train, y_train

    def preprocess_audio(self, audio_data, sr=None):
        """
        Preprocess audio for model input, with enhancements for speech impediments.

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

        # Perform cleanup for speech impediment enhancement
        # 1. Emphasis filter - enhance higher frequencies for consonant clarity
        emphasized_audio = self.apply_emphasis_filter(audio_data)

        # 2. Noise reduction and speech enhancement preprocessing
        emphasized_audio = self.denoise_speech(emphasized_audio)

        # 3. Dynamic range compression to improve quieter consonants
        emphasized_audio = self.compress_dynamic_range(emphasized_audio)

        # 4. Formant enhancement to make speech more defined
        emphasized_audio = self.enhance_formants(emphasized_audio)

        # Compute mel spectrogram from enhanced audio
        mel_spec = librosa.feature.melspectrogram(
            y=emphasized_audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1] range for sigmoid activation in the model output
        normalized_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)

        # Ensure proper shape
        normalized_spec = np.expand_dims(normalized_spec, axis=-1)

        return normalized_spec

    # Existing audio processing methods remain unchanged
    # ...

    def save_model(self, user_specific=True):
        """
        Save the current model with proper extension.

        Args:
            user_specific (bool): Whether to save as user-specific model

        Returns:
            bool: True if saved successfully, False otherwise
        """
        if self.model is None:
            logger.error("No model to save")
            return False

        model_path = self.user_model_path if user_specific else self.base_model_path

        # Add the .keras extension to the model path
        if not model_path.endswith('.keras') and not model_path.endswith('.h5'):
            model_path = model_path + '.keras'

        try:
            self.model.save(model_path)
            logger.info(f"Saved model to {model_path}")

            # Also save model architecture visualization
            try:
                dot_img_file = model_path + '.png'
                tf.keras.utils.plot_model(
                    self.model,
                    to_file=dot_img_file,
                    show_shapes=True,
                    show_layer_names=True
                )
                logger.info(f"Saved model visualization to {dot_img_file}")
            except Exception as viz_error:
                logger.warning(f"Could not save model visualization: {viz_error}")

            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def get_enhancement_statistics(self):
        """
        Get statistics about model enhancement performance.

        Returns:
            dict: Enhancement statistics
        """
        stats = self.enhancement_metrics.copy()

        # Add confidence ratio
        if stats["total_processed"] > 0:
            stats["high_confidence_ratio"] = stats["high_confidence_count"] / stats["total_processed"]
        else:
            stats["high_confidence_ratio"] = 0.0

        # Add average processing time
        if len(stats["processing_times_ms"]) > 0:
            stats["avg_processing_time_ms"] = sum(stats["processing_times_ms"]) / len(stats["processing_times_ms"])
        else:
            stats["avg_processing_time_ms"] = 0.0

        # Remove the raw list of processing times to keep the response compact
        del stats["processing_times_ms"]

        # Add model type
        stats["model_type"] = "user" if self.user_id != "default" else "base"
        stats["user_id"] = self.user_id

        return stats