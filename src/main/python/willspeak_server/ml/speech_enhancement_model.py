"""
Enhanced Speech Enhancement Model for WillSpeak

This module provides an improved TensorFlow model for enhancing speech clarity
specifically tailored to users with speech impediments.
"""
import gc
import os
import numpy as np
import tensorflow as tf
from keras.src.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate,
    Dropout, BatchNormalization, Reshape, Dense, LSTM, Bidirectional,
    Permute, Activation, Lambda, Add, Multiply, AveragePooling2D,
    LeakyReLU, GRU, TimeDistributed, Attention
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import librosa
from tensorflow.keras import backend as K
import librosa.display
import soundfile as sf
from datetime import datetime
import json
import matplotlib.pyplot as plt
from loguru import logger
import sys
from pathlib import Path
import scipy.signal as signal


class SpeechEnhancementModel:
    """
    An advanced deep learning model for enhancing speech clarity.

    This model is specifically designed to handle speech with various impediments,
    including slurred speech and limited muscle control issues.
    """

    def __init__(self, user_id=None, model_dir=None):
        """
        Initialize the enhanced speech enhancement model with improved parameters.
        """
        self.user_id = user_id or "default"
        self.model_dir = model_dir or os.path.join(os.getcwd(), "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Optimized model paths
        self.base_model_path = os.path.join(self.model_dir, "base_model")
        self.user_model_path = os.path.join(self.model_dir, f"user_{self.user_id}_model")

        # Adjusted model parameters for better temporal resolution
        self.sr = 16000
        self.n_fft = 1024  # Increased to ensure sufficient frequency resolution
        self.hop_length = 256
        self.n_mels = 128  # Match model input height
        self.time_steps = 128  # Match model input width
        self.min_audio_length = self.hop_length * (self.time_steps - 1)  # Minimum required samples

        # Enhanced formant and consonant parameters
        self.formant_emphasis = 2.0  # Stronger formant boost
        self.formant_bandwidth = 125  # Narrower bandwidth for precision
        self.consonant_emphasis = 2.5  # Stronger consonant boost
        self.high_freq_gain = 1.8  # Increased high-frequency gain

        # Model and buffers
        self.model = None
        self.audio_buffer = np.zeros(0)
        self.processed_buffer = np.zeros(0)
        self.buffer_size = int(self.sr * 0.3)  # 300ms buffer for lower latency
        self.overlap_size = int(self.buffer_size * 0.5)  # 50% overlap for smooth stitching

        # Formant tracking with adaptive history
        self.formant_history = []
        self.max_formant_history = 15  # Longer history for stability

        # Performance metrics
        self.enhancement_metrics = {
            "total_processed": 0,
            "avg_confidence": 0.0,
            "high_confidence_count": 0,
            "processing_times_ms": []
        }

        # Store normalization parameters
        self.normalization_params = {}

        logger.info(f"Initialized EnhancedSpeechModel for user: {self.user_id}")

    def build_model(self, input_shape=(128, 128, 1)):
        """
        Build an improved speech model architecture with attention mechanisms
        and specialized speech enhancement components.

        Args:
            input_shape (tuple): Shape of input spectrogram

        Returns:
            tf.keras.Model: The compiled model
        """
        logger.info(f"Building improved speech enhancement model with input shape: {input_shape}")

        # Input layer
        inputs = Input(shape=input_shape, name="input_spectrogram")

        # -- ENCODER PATH with Speech-Specific Attention --

        # First level - Extract basic features
        conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # Second level - Focus on speech components
        conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        conv2 = BatchNormalization()(conv2)
        # Add speech-focused attention
        attention2 = self.speech_attention_block(conv2, 64)
        pool2 = MaxPooling2D(pool_size=(2, 2))(attention2)

        # Third level - Enhanced feature extraction
        conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        conv3 = BatchNormalization()(conv3)
        # Add formant-aware attention
        attention3 = self.formant_attention_block(conv3, 128)
        pool3 = MaxPooling2D(pool_size=(2, 2))(attention3)

        # Fourth level - Deep feature extraction
        conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        conv4 = BatchNormalization()(conv4)
        # Add consonant-focused attention
        attention4 = self.consonant_attention_block(conv4, 256)
        pool4 = MaxPooling2D(pool_size=(2, 2))(attention4)

        # -- BOTTLENECK with multi-scale feature integration --
        # Using multi-dilated convolutions to capture different speech features
        bottleneck1 = Conv2D(512, (3, 3), dilation_rate=(1, 1), padding='same')(pool4)
        bottleneck1 = LeakyReLU(alpha=0.2)(bottleneck1)
        bottleneck1 = BatchNormalization()(bottleneck1)

        bottleneck2 = Conv2D(512, (3, 3), dilation_rate=(2, 2), padding='same')(pool4)
        bottleneck2 = LeakyReLU(alpha=0.2)(bottleneck2)
        bottleneck2 = BatchNormalization()(bottleneck2)

        bottleneck3 = Conv2D(512, (3, 3), dilation_rate=(4, 4), padding='same')(pool4)
        bottleneck3 = LeakyReLU(alpha=0.2)(bottleneck3)
        bottleneck3 = BatchNormalization()(bottleneck3)

        # Concatenate multi-scale features
        bottleneck = Concatenate()([bottleneck1, bottleneck2, bottleneck3])
        bottleneck = Conv2D(512, (1, 1), padding='same')(bottleneck)  # 1x1 conv to reduce channels
        bottleneck = LeakyReLU(alpha=0.2)(bottleneck)
        bottleneck = BatchNormalization()(bottleneck)
        bottleneck = Dropout(0.3)(bottleneck)

        # -- DECODER PATH with enhanced skip connections --

        # Level 4 up - Focus on reconstructing high-level speech features
        up4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(bottleneck)
        up4 = LeakyReLU(alpha=0.2)(up4)
        # Enhanced skip connection with feature selection
        skip4 = self.feature_gate_block(attention4, up4)
        merge4 = Concatenate()([up4, skip4])
        deconv4 = Conv2D(256, (3, 3), padding='same')(merge4)
        deconv4 = LeakyReLU(alpha=0.2)(deconv4)
        deconv4 = BatchNormalization()(deconv4)
        deconv4 = Conv2D(256, (3, 3), padding='same')(deconv4)
        deconv4 = LeakyReLU(alpha=0.2)(deconv4)
        deconv4 = BatchNormalization()(deconv4)

        # Level 3 up - Focus on reconstructing formants
        up3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(deconv4)
        up3 = LeakyReLU(alpha=0.2)(up3)
        # Enhanced skip connection
        skip3 = self.feature_gate_block(attention3, up3)
        merge3 = Concatenate()([up3, skip3])
        deconv3 = Conv2D(128, (3, 3), padding='same')(merge3)
        deconv3 = LeakyReLU(alpha=0.2)(deconv3)
        deconv3 = BatchNormalization()(deconv3)
        deconv3 = Conv2D(128, (3, 3), padding='same')(deconv3)
        deconv3 = LeakyReLU(alpha=0.2)(deconv3)
        deconv3 = BatchNormalization()(deconv3)

        # Level 2 up - Focus on phoneme transitions
        up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(deconv3)
        up2 = LeakyReLU(alpha=0.2)(up2)
        # Enhanced skip connection
        skip2 = self.feature_gate_block(attention2, up2)
        merge2 = Concatenate()([up2, skip2])
        deconv2 = Conv2D(64, (3, 3), padding='same')(merge2)
        deconv2 = LeakyReLU(alpha=0.2)(deconv2)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Conv2D(64, (3, 3), padding='same')(deconv2)
        deconv2 = LeakyReLU(alpha=0.2)(deconv2)
        deconv2 = BatchNormalization()(deconv2)

        # Level 1 up - Return to original size with speech clarity enhancements
        up1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(deconv2)
        up1 = LeakyReLU(alpha=0.2)(up1)
        # Basic skip connection
        merge1 = Concatenate()([up1, conv1])
        deconv1 = Conv2D(32, (3, 3), padding='same')(merge1)
        deconv1 = LeakyReLU(alpha=0.2)(deconv1)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Conv2D(32, (3, 3), padding='same')(deconv1)
        deconv1 = LeakyReLU(alpha=0.2)(deconv1)
        deconv1 = BatchNormalization()(deconv1)

        # Final speech enhancement block with special focus on intelligibility
        enhanced = self.intelligibility_enhancement_block(deconv1)

        # Output layer - using tanh activation for better dynamic range control
        outputs = Conv2D(1, (1, 1), activation='tanh', name="enhanced_spectrogram")(enhanced)

        # Create model
        model = Model(inputs, outputs)
        self.model = model  # Assign the model to self.model first

        # Define custom speech intelligibility loss function
        def speech_intelligibility_loss(y_true, y_pred):
            # Basic mean squared error component
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

            # Spectral loss with emphasis on speech-critical frequency bands
            y_true_flat = tf.reshape(y_true, [-1, input_shape[0], input_shape[1]])
            y_pred_flat = tf.reshape(y_pred, [-1, input_shape[0], input_shape[1]])

            # Create a weighting mask that emphasizes speech-critical frequencies
            # Consonant range (higher frequencies)
            consonant_range = np.ones(input_shape[0])
            consonant_start = int(input_shape[0] * 0.6)  # 60% and above (higher frequencies)
            consonant_range[consonant_start:] = 2.5

            # Vowel formant range (mid frequencies for vowel intelligibility)
            formant_start = int(input_shape[0] * 0.15)
            formant_end = int(input_shape[0] * 0.6)
            consonant_range[formant_start:formant_end] = 2.0

            # Convert to tensor
            freq_weights = tf.constant(consonant_range, dtype=tf.float32)
            freq_weights = tf.reshape(freq_weights, [input_shape[0], 1])

            # Apply frequency weighting
            weighted_true = y_true_flat * freq_weights
            weighted_pred = y_pred_flat * freq_weights

            # Spectral loss with speech-focused weighting
            spectral_loss = tf.reduce_mean(tf.abs(weighted_true - weighted_pred))

            # Temporal continuity loss to preserve transitions between phonemes
            # Calculate temporal gradient using diff
            true_temporal = y_true_flat[:, 1:, :] - y_true_flat[:, :-1, :]
            pred_temporal = y_pred_flat[:, 1:, :] - y_pred_flat[:, :-1, :]

            # Measure temporal continuity differences
            temporal_loss = tf.reduce_mean(tf.abs(true_temporal - pred_temporal))

            # Combined loss with higher weights on speech-critical components
            return 0.5 * mse_loss + 1.0 * spectral_loss + 0.5 * temporal_loss

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

        # Custom metric for consonant preservation
        def consonant_preservation_metric(y_true, y_pred):
            # Reshape to get the spectrogram format
            y_true_spec = tf.reshape(y_true, [-1, input_shape[0], input_shape[1]])
            y_pred_spec = tf.reshape(y_pred, [-1, input_shape[0], input_shape[1]])

            # Focus on higher frequencies (consonants)
            high_freq_start = int(input_shape[0] * 0.6)
            true_high = y_true_spec[:, high_freq_start:, :]
            pred_high = y_pred_spec[:, high_freq_start:, :]

            # Calculate correlation for high frequencies
            true_high_flat = tf.reshape(true_high, [-1])
            pred_high_flat = tf.reshape(pred_high, [-1])

            # Calculate mean values
            true_high_mean = tf.reduce_mean(true_high_flat)
            pred_high_mean = tf.reduce_mean(pred_high_flat)

            # Center values
            true_high_centered = true_high_flat - true_high_mean
            pred_high_centered = pred_high_flat - pred_high_mean

            # Calculate correlation
            numerator = tf.reduce_sum(true_high_centered * pred_high_centered)
            denominator = tf.sqrt(tf.reduce_sum(tf.square(true_high_centered)) *
                                  tf.reduce_sum(tf.square(pred_high_centered)))

            # Avoid division by zero
            denominator = tf.maximum(denominator, 1e-12)
            high_freq_correlation = numerator / denominator

            return high_freq_correlation

        initial_lr = 0.001
        scaled_lr = initial_lr
        self.model.compile(
            optimizer=Adam(learning_rate=scaled_lr),
            loss=speech_intelligibility_loss,
            metrics=['mean_absolute_error', speech_clarity_metric, consonant_preservation_metric]
        )

        model.summary(print_fn=logger.info)
        return model

    def speech_attention_block(self, x, filters):
        """
        Create an attention block specialized for speech features.
        Helps focus on important speech components like formants and consonants.

        Args:
            x: Input tensor
            filters: Number of filters

        Returns:
            Tensor with speech-focused attention applied
        """
        # Channel attention with speech frequency bias
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
        avg_pool = tf.keras.layers.Reshape((1, 1, filters))(avg_pool)
        avg_pool = tf.keras.layers.Conv2D(filters // 4, kernel_size=1, activation='relu')(avg_pool)
        avg_pool = tf.keras.layers.Conv2D(filters, kernel_size=1, activation='sigmoid')(avg_pool)

        # Apply channel attention
        channel_attention = tf.keras.layers.Multiply()([x, avg_pool])

        # Define Lambda functions properly with imported tf
        def reduce_mean_func(x):
            return tf.reduce_mean(x, axis=-1, keepdims=True)

        def reduce_max_func(x):
            return tf.reduce_max(x, axis=-1, keepdims=True)

        # Spatial attention with focus on formant regions
        avg_pool_spatial = tf.keras.layers.Lambda(reduce_mean_func)(x)
        max_pool_spatial = tf.keras.layers.Lambda(reduce_max_func)(x)

        concat = tf.keras.layers.Concatenate()([avg_pool_spatial, max_pool_spatial])
        # Larger kernel for better speech pattern awareness
        spatial_attention = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)

        # Apply spatial attention
        spatial_attention_output = tf.keras.layers.Multiply()([channel_attention, spatial_attention])

        # Add residual connection
        return tf.keras.layers.Add()([x, spatial_attention_output])

    def formant_attention_block(self, x, filters):
        """
        Create an attention block specialized for formant enhancement.
        Focuses on vowel sounds and voiced regions.

        Args:
            x: Input tensor
            filters: Number of filters

        Returns:
            Tensor with formant-focused attention applied
        """
        # Create a formant-focused feature map
        formant_feature = tf.keras.layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(x)

        # Add frequency-focused attention (formants are in mid-frequencies)
        # Create a bias toward formant frequency ranges
        formant_bias = tf.keras.layers.Conv2D(filters, (1, 5), padding='same')(x)
        formant_bias = tf.keras.layers.LeakyReLU(0.2)(formant_bias)

        # Combine with original features
        formant_enhanced = tf.keras.layers.Add()([formant_feature, formant_bias])
        formant_gate = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', activation='sigmoid')(formant_enhanced)

        # Apply gating
        gated_output = tf.keras.layers.Multiply()([x, formant_gate])

        # Add residual connection
        return tf.keras.layers.Add()([x, gated_output])

    def consonant_attention_block(self, x, filters):
        """
        Create an attention block specialized for consonant enhancement.
        Focuses on high frequency and rapid transitions.

        Args:
            x: Input tensor
            filters: Number of filters

        Returns:
            Tensor with consonant-focused attention applied
        """
        # Create a consonant-focused feature map (smaller kernel for detail)
        consonant_feature = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)

        # Add time-focused attention (consonants have rapid changes)
        time_feature = tf.keras.layers.Conv2D(filters, (3, 1), padding='same')(x)
        time_feature = tf.keras.layers.LeakyReLU(0.2)(time_feature)

        # High-frequency focus
        freq_feature = tf.keras.layers.Conv2D(filters, (1, 3), padding='same')(x)
        freq_feature = tf.keras.layers.LeakyReLU(0.2)(freq_feature)

        # Combine features
        combined = tf.keras.layers.Add()([consonant_feature, time_feature, freq_feature])
        attention_map = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', activation='sigmoid')(combined)

        # Apply attention
        enhanced = tf.keras.layers.Multiply()([x, attention_map])

        # Add residual connection
        return tf.keras.layers.Add()([x, enhanced])

    def feature_gate_block(self, skip_features, up_features):
        """
        Create a gating mechanism for skip connections to select relevant features.
        Helps control information flow for better speech reconstruction.

        Args:
            skip_features: Features from encoder path
            up_features: Features from decoder path

        Returns:
            Gated skip features
        """
        # Get number of filters from the input tensor
        filters = skip_features.shape[-1]

        # Create gate from upsampled features
        gate = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(up_features)
        gate = tf.keras.layers.BatchNormalization()(gate)
        gate = tf.keras.layers.Activation('sigmoid')(gate)

        # Apply gate to skip features
        gated_skip = tf.keras.layers.Multiply()([skip_features, gate])

        return gated_skip

    def intelligibility_enhancement_block(self, x):
        """
        Final processing block specifically designed to enhance speech intelligibility.
        Applies targeted enhancements for vowel formants and consonants.

        Args:
            x: Input tensor

        Returns:
            Enhanced tensor with improved speech intelligibility features
        """
        # Extract features of different types
        filters = x.shape[-1]

        # Enhance formant structure (using dilated convolutions)
        formant_feature = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=(2, 2))(x)
        formant_feature = tf.keras.layers.LeakyReLU(0.2)(formant_feature)

        # Enhance consonant detail (using small kernel)
        consonant_feature = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        consonant_feature = tf.keras.layers.LeakyReLU(0.2)(consonant_feature)

        # Enhance temporal transitions (critical for intelligibility)
        transition_feature = tf.keras.layers.Conv2D(filters, (5, 1), padding='same')(x)
        transition_feature = tf.keras.layers.LeakyReLU(0.2)(transition_feature)

        # Combine all enhancements
        combined = tf.keras.layers.Concatenate()([x, formant_feature, consonant_feature, transition_feature])

        # Apply final integration
        output = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(combined)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.LeakyReLU(0.2)(output)

        return output

    def spectral_loss(self, y_true, y_pred, input_shape):
        """Weighted spectral loss emphasizing speech-critical bands."""
        freq_weights = tf.ones([input_shape[0], 1], dtype=tf.float32)
        freq_weights = freq_weights * tf.cast(tf.linspace(1.0, 2.5, input_shape[0])[:, None], dtype=tf.float32)  # Gradual emphasis
        return tf.reduce_mean(tf.abs(y_true - y_pred) * freq_weights)

    def temporal_loss(self, y_true, y_pred):
        """Temporal continuity loss."""
        true_diff = y_true[:, 1:, :] - y_true[:, :-1, :]
        pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        return tf.reduce_mean(tf.abs(true_diff - pred_diff))

    def perceptual_loss(self, y_true, y_pred):
        """Simple perceptual loss using spectrogram correlation."""
        true_flat = tf.reshape(y_true, [-1])
        pred_flat = tf.reshape(y_pred, [-1])
        return 1.0 - tf.reduce_mean(tf.abs(true_flat - pred_flat))  # Simplified for now

    def load_model(self, user_specific=True):
        """
        Load a pre-trained model with proper extension handling and Lambda layer support.

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

        # Enable unsafe deserialization for Lambda layers
        tf.keras.config.enable_unsafe_deserialization()

        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                    logger.info(f"Attempting to load model from {model_path}")
                    self.model = load_model(model_path, compile=False)

                    # Recompile the model with custom loss and metrics
                    def speech_intelligibility_loss(y_true, y_pred):
                        # Basic mean squared error component
                        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

                        # Reshape inputs for spectral loss
                        y_true_flat = tf.reshape(y_true, [-1, 128, 128])
                        y_pred_flat = tf.reshape(y_pred, [-1, 128, 128])

                        # Create a weighting mask that emphasizes speech-critical frequencies
                        # Consonant range (higher frequencies)
                        consonant_range = np.ones(128)
                        consonant_start = int(128 * 0.6)  # 60% and above (higher frequencies)
                        consonant_range[consonant_start:] = 2.5

                        # Vowel formant range (mid frequencies for vowel intelligibility)
                        formant_start = int(128 * 0.15)
                        formant_end = int(128 * 0.6)
                        consonant_range[formant_start:formant_end] = 2.0

                        # Convert to tensor
                        freq_weights = tf.constant(consonant_range, dtype=tf.float32)
                        freq_weights = tf.reshape(freq_weights, [128, 1])

                        # Apply frequency weighting
                        weighted_true = y_true_flat * freq_weights
                        weighted_pred = y_pred_flat * freq_weights

                        # Spectral loss with speech-focused weighting
                        spectral_loss = tf.reduce_mean(tf.abs(weighted_true - weighted_pred))

                        # Temporal continuity loss to preserve transitions between phonemes
                        # Calculate temporal gradient using diff
                        true_temporal = y_true_flat[:, 1:, :] - y_true_flat[:, :-1, :]
                        pred_temporal = y_pred_flat[:, 1:, :] - y_pred_flat[:, :-1, :]

                        # Measure temporal continuity differences
                        temporal_loss = tf.reduce_mean(tf.abs(true_temporal - pred_temporal))

                        # Combined loss with higher weights on speech-critical components
                        return 0.5 * mse_loss + 1.0 * spectral_loss + 0.5 * temporal_loss

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

                    def consonant_preservation_metric(y_true, y_pred):
                        # Reshape to get the spectrogram format
                        y_true_spec = tf.reshape(y_true, [-1, 128, 128])
                        y_pred_spec = tf.reshape(y_pred, [-1, 128, 128])

                        # Focus on higher frequencies (consonants)
                        high_freq_start = int(128 * 0.6)
                        true_high = y_true_spec[:, high_freq_start:, :]
                        pred_high = y_pred_spec[:, high_freq_start:, :]

                        # Calculate correlation for high frequencies
                        true_high_flat = tf.reshape(true_high, [-1])
                        pred_high_flat = tf.reshape(pred_high, [-1])

                        # Calculate mean values
                        true_high_mean = tf.reduce_mean(true_high_flat)
                        pred_high_mean = tf.reduce_mean(pred_high_flat)

                        # Center values
                        true_high_centered = true_high_flat - true_high_mean
                        pred_high_centered = pred_high_flat - pred_high_mean

                        # Calculate correlation
                        numerator = tf.reduce_sum(true_high_centered * pred_high_centered)
                        denominator = tf.sqrt(tf.reduce_sum(tf.square(true_high_centered)) *
                                              tf.reduce_sum(tf.square(pred_high_centered)))

                        # Avoid division by zero
                        denominator = tf.maximum(denominator, 1e-12)
                        high_freq_correlation = numerator / denominator

                        return high_freq_correlation

                    # Recompile the model
                    self.model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss=speech_intelligibility_loss,
                        metrics=['mean_absolute_error', speech_clarity_metric, consonant_preservation_metric]
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

    def prepare_training_data(self, unclear_audio_paths, clear_audio_paths):
        """
        Enhanced function to prepare training data from audio file pairs with
        sophisticated preprocessing and validation.

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
        error_count = 0
        warnings_count = 0

        # Create a default shape for empty cases
        default_shape = (self.n_mels, self.time_steps, 1)

        # Track file statistics for reporting
        file_stats = {
            'total_duration': 0,
            'min_duration': float('inf'),
            'max_duration': 0,
            'avg_amplitude_unclear': [],
            'avg_amplitude_clear': [],
            'noise_levels': []
        }

        # Process in batches to improve efficiency
        batch_size = min(20, total_files)  # Process up to 20 files at a time
        num_batches = (total_files + batch_size - 1) // batch_size

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, total_files)

            logger.info(f"Processing batch {batch + 1}/{num_batches}, files {start_idx + 1}-{end_idx}")

            batch_unclear_specs = []
            batch_clear_specs = []

            for i in range(start_idx, end_idx):
                unclear_path = unclear_audio_paths[i]
                clear_path = clear_audio_paths[i]

                try:
                    # Check if files exist
                    if not os.path.exists(unclear_path) or not os.path.exists(clear_path):
                        logger.error(f"File not found for pair {i + 1}: {unclear_path} or {clear_path}")
                        error_count += 1
                        continue

                    # Load audio files with robust error handling
                    try:
                        unclear_audio, sr1 = librosa.load(unclear_path, sr=self.sr)
                        clear_audio, sr2 = librosa.load(clear_path, sr=self.sr)
                    except Exception as e:
                        logger.error(f"Error loading audio files for pair {i + 1}: {str(e)}")
                        error_count += 1
                        continue

                    # Verify audio data is valid
                    if len(unclear_audio) < 100 or len(clear_audio) < 100:
                        logger.warning(f"Audio files too short for pair {i + 1}, skipping")
                        warnings_count += 1
                        continue

                    # Track audio statistics
                    file_stats['total_duration'] += len(unclear_audio) / self.sr
                    file_stats['min_duration'] = min(file_stats['min_duration'], len(unclear_audio) / self.sr)
                    file_stats['max_duration'] = max(file_stats['max_duration'], len(unclear_audio) / self.sr)
                    file_stats['avg_amplitude_unclear'].append(np.mean(np.abs(unclear_audio)))
                    file_stats['avg_amplitude_clear'].append(np.mean(np.abs(clear_audio)))

                    # Estimate noise level (using simplified SNR calculation)
                    # Fixed: Handle different lengths properly when estimating noise
                    min_len = min(len(unclear_audio), len(clear_audio))
                    noise_estimate = np.std(unclear_audio[:min_len] - clear_audio[:min_len]) / (
                                np.std(clear_audio[:min_len]) + 1e-8)
                    file_stats['noise_levels'].append(noise_estimate)

                    # *** Key Fix: Make both audio files the same length ***
                    # Resample both to same length if they differ
                    if len(unclear_audio) != len(clear_audio):
                        logger.warning(
                            f"Length mismatch in pair {i + 1}: unclear={len(unclear_audio)}, clear={len(clear_audio)}")

                        # Resample the longer one to match the shorter one's length
                        if len(unclear_audio) > len(clear_audio):
                            # Resample unclear to match clear
                            target_length = len(clear_audio)
                            unclear_audio = librosa.resample(
                                unclear_audio,
                                orig_sr=self.sr,
                                target_sr=int(self.sr * target_length / len(unclear_audio))
                            )
                        else:
                            # Resample clear to match unclear
                            target_length = len(unclear_audio)
                            clear_audio = librosa.resample(
                                clear_audio,
                                orig_sr=self.sr,
                                target_sr=int(self.sr * target_length / len(clear_audio))
                            )

                        logger.info(f"Resampled audio lengths: unclear={len(unclear_audio)}, clear={len(clear_audio)}")

                    # Segment long audio files into multiple training samples
                    # This maximizes usable data and ensures consistent segment length
                    max_segment_len = int(4.0 * self.sr)  # 4 seconds max per segment
                    min_segment_len = int(1.0 * self.sr)  # 1 second min per segment

                    # Skip very short audio samples
                    if len(unclear_audio) < min_segment_len:
                        logger.warning(f"Skipping pair {i + 1} - audio too short: {len(unclear_audio) / self.sr:.2f}s")
                        warnings_count += 1
                        continue

                    # Segment longer files
                    if len(unclear_audio) > max_segment_len:
                        segments = []
                        for start in range(0, len(unclear_audio) - min_segment_len, min_segment_len):
                            end = min(start + max_segment_len, len(unclear_audio))
                            if end - start >= min_segment_len:
                                segments.append((start, end))
                    else:
                        segments = [(0, len(unclear_audio))]

                    # Process each segment
                    for seg_idx, (start, end) in enumerate(segments):
                        try:
                            # Extract segment
                            seg_unclear = unclear_audio[start:end]
                            seg_clear = clear_audio[start:end]

                            # Skip segments with very low energy
                            if np.mean(np.abs(seg_unclear)) < 0.005 or np.mean(np.abs(seg_clear)) < 0.005:
                                continue

                            # Process audio into spectrograms with robust error handling
                            try:
                                # Convert to spectrograms
                                unclear_spec = self.to_spectrogram(seg_unclear)
                                clear_spec = self.to_spectrogram(seg_clear)

                                # Verify shapes match after processing
                                if unclear_spec.shape != clear_spec.shape:
                                    logger.warning(f"Spectrogram shape mismatch in pair {i + 1}, segment {seg_idx}: "
                                                   f"unclear={unclear_spec.shape}, clear={clear_spec.shape}")
                                    continue

                            except Exception as e:
                                logger.error(
                                    f"Error in spectrogram conversion for pair {i + 1}, segment {seg_idx}: {str(e)}")
                                continue

                            # Validate spectrograms for variance and consistency
                            if self._validate_spectrogram_pair(unclear_spec, clear_spec):
                                batch_unclear_specs.append(unclear_spec)
                                batch_clear_specs.append(clear_spec)
                                valid_pairs += 1
                                if seg_idx > 0:
                                    logger.info(f"Successfully processed pair {i + 1}, segment {seg_idx + 1}")
                                else:
                                    logger.info(f"Successfully processed pair {i + 1}")
                            else:
                                logger.warning(f"Spectrograms failed validation for pair {i + 1}, segment {seg_idx}")

                        except Exception as e:
                            logger.error(f"Error processing segment {seg_idx} for pair {i + 1}: {str(e)}")
                            continue

                except Exception as e:
                    logger.error(f"Error processing audio pair {i + 1}: {str(e)}")
                    error_count += 1
                    continue

            # Add batch to total collection
            unclear_specs.extend(batch_unclear_specs)
            clear_specs.extend(batch_clear_specs)

            # Periodic memory cleanup
            gc.collect()

        # Report data preparation statistics
        if file_stats['avg_amplitude_unclear']:
            file_stats['avg_amplitude_unclear'] = np.mean(file_stats['avg_amplitude_unclear'])
            file_stats['avg_amplitude_clear'] = np.mean(file_stats['avg_amplitude_clear'])
            file_stats['avg_noise_level'] = np.mean(file_stats['noise_levels']) if file_stats['noise_levels'] else 0

        logger.info(f"Data preparation completed: {valid_pairs} valid pairs/segments, "
                    f"{error_count} errors, {warnings_count} warnings")

        logger.info(f"Audio statistics: {valid_pairs} segments, "
                    f"total duration: {file_stats['total_duration']:.2f}s, "
                    f"min: {file_stats['min_duration']:.2f}s, "
                    f"max: {file_stats['max_duration']:.2f}s")

        if len(unclear_specs) == 0:
            logger.warning("No valid training pairs found. Creating minimal recovery dataset.")
            # Create a minimal recovery dataset to prevent errors
            dummy_spec = np.random.normal(0.5, 0.1, default_shape)
            unclear_specs = [dummy_spec]
            clear_specs = [dummy_spec.copy()]  # Use similar specs for this dummy case
            valid_pairs = 1
            logger.warning("Added dummy training data to prevent complete failure")

        # Convert to numpy arrays
        X_train = np.array(unclear_specs)
        y_train = np.array(clear_specs)

        logger.info(f"Final training data shapes: X={X_train.shape}, Y={y_train.shape}")

        return X_train, y_train

    def to_spectrogram(self, audio):
        """
        Convert audio to a fixed-size mel spectrogram matching model input.
        Includes enhanced error handling and validation.
        """
        try:
            # Ensure we have valid audio
            if not isinstance(audio, np.ndarray):
                logger.warning("Input to to_spectrogram is not a numpy array")
                audio = np.array(audio, dtype=np.float32)

            if len(audio) == 0:
                logger.warning("Empty audio provided to to_spectrogram")
                # Return zero-filled spectrogram
                return np.zeros((self.n_mels, self.time_steps, 1))

            # Pad if needed
            if len(audio) < self.min_audio_length:
                audio = np.pad(audio, (0, self.min_audio_length - len(audio)), mode='constant')

            # Ensure audio is long enough for the desired time steps
            spec = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, window='hann')

            # Verify we got a valid spectrogram
            if spec.size == 0 or np.isnan(spec).any():
                logger.warning("STFT produced invalid spectrogram, using zeros")
                return np.zeros((self.n_mels, self.time_steps, 1))

            mel_spec = librosa.feature.melspectrogram(S=np.abs(spec) ** 2, sr=self.sr, n_mels=self.n_mels)

            # Handle potential NaN values
            if np.isnan(mel_spec).any():
                logger.warning("NaN values in mel spectrogram, replacing with zeros")
                mel_spec = np.nan_to_num(mel_spec)

            # Handle zero or negative values before log
            if np.min(mel_spec) <= 0:
                min_positive = np.min(mel_spec[mel_spec > 0]) if np.any(mel_spec > 0) else 1e-6
                mel_spec = np.maximum(mel_spec, min_positive)

            mel_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Trim or pad to fixed time steps (128)
            if mel_db.shape[1] > self.time_steps:
                mel_db = mel_db[:, :self.time_steps]
            elif mel_db.shape[1] < self.time_steps:
                mel_db = np.pad(mel_db, ((0, 0), (0, self.time_steps - mel_db.shape[1])), mode='constant')

            # Normalize to [0,1] range
            mel_min, mel_max = np.min(mel_db), np.max(mel_db)
            if mel_max > mel_min:
                mel_db = (mel_db - mel_min) / (mel_max - mel_min)
            else:
                # If min == max, set to 0.5 to avoid division by zero
                mel_db = np.ones_like(mel_db) * 0.5

            return mel_db[..., np.newaxis]  # Shape: (128, 128, 1)

        except Exception as e:
            logger.error(f"Error in to_spectrogram: {str(e)}")
            # Return zero-filled spectrogram on error
            return np.zeros((self.n_mels, self.time_steps, 1))

    def _validate_spectrogram_pair(self, unclear_spec, clear_spec):
        """
        Validates a pair of spectrograms for training quality.

        Args:
            unclear_spec: Input spectrogram (impaired speech)
            clear_spec: Target spectrogram (clear speech)

        Returns:
            bool: True if the pair is valid for training
        """
        # 1. Check shapes match
        if unclear_spec.shape != clear_spec.shape:
            return False

        # 2. Check for reasonable variance in both specs
        unclear_var = np.var(unclear_spec)
        clear_var = np.var(clear_spec)

        min_variance = 0.001
        if unclear_var < min_variance or clear_var < min_variance:
            return False

        # 3. Check for NaN or inf values
        if np.isnan(unclear_spec).any() or np.isnan(clear_spec).any():
            return False

        if np.isinf(unclear_spec).any() or np.isinf(clear_spec).any():
            return False

        # 4. Check that specs are different but correlated (should be similar but not identical)
        # Flatten for correlation calculation
        unclear_flat = unclear_spec.flatten()
        clear_flat = clear_spec.flatten()

        # Calculate correlation
        correlation = np.corrcoef(unclear_flat, clear_flat)[0, 1]

        # Correlation should be positive but not too high (not identical)
        if correlation < 0.3 or correlation > 0.99:
            return False

        # 5. Check for sufficient spectral content (not just silence or noise)
        # Calculate energy in speech-critical bands
        formant_region = slice(int(unclear_spec.shape[0] * 0.1), int(unclear_spec.shape[0] * 0.6))
        formant_energy_unclear = np.mean(unclear_spec[formant_region, :, :])
        formant_energy_clear = np.mean(clear_spec[formant_region, :, :])

        # Ensure there's reasonable energy in speech regions
        min_energy = 0.1
        if formant_energy_unclear < min_energy or formant_energy_clear < min_energy:
            return False

        # All checks passed
        return True

    def train(self, train_data, test_data=None, epochs=50, batch_size=32, callbacks=None):
        """
        Improved training algorithm with advanced data handling, adaptive learning,
        and enhanced optimization for speech enhancement models.

        Args:
            train_data (tuple): Tuple containing (X_train, y_train) with input and target spectrograms
            test_data (tuple, optional): Tuple containing (X_test, y_test) for validation
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            callbacks (list, optional): Additional callbacks for training

        Returns:
            dict: Training history with metrics
        """
        try:
            logger.info(
                f"Starting improved training with data shapes: X={train_data[0].shape}, y={train_data[1].shape}")

            # Verify input data integrity
            if not isinstance(train_data, tuple) or len(train_data) != 2:
                raise ValueError(f"train_data must be a tuple of (inputs, targets), got {type(train_data)}")

            # Build model if not already built
            if self.model is None:
                if len(train_data[0].shape) < 3:
                    logger.error(f"Invalid input shape: {train_data[0].shape}, expected at least 3 dimensions")
                    input_shape = (128, 128, 1)  # Use default shape
                    logger.info(f"Using default input shape: {input_shape}")
                else:
                    input_shape = train_data[0].shape[1:]
                self.build_model(input_shape)

            # Verify data dimensions
            num_samples = train_data[0].shape[0]
            if num_samples == 0:
                logger.error("No training samples provided")
                raise ValueError("Training data contains 0 samples")

            if train_data[1].shape[0] != num_samples:
                logger.error(
                    f"Input and target sample counts don't match: {train_data[0].shape[0]} vs {train_data[1].shape[0]}")
                raise ValueError(f"Invalid training data: {train_data[0].shape} and {train_data[1].shape} mismatch")

            # Setup basic callbacks if none provided
            if callbacks is None:
                callbacks = []

                # Add early stopping
                callbacks.append(EarlyStopping(
                    monitor='loss',
                    patience=10,
                    restore_best_weights=True
                ))

                # Add learning rate reduction on plateau
                callbacks.append(ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ))

                # Add model checkpoint to save best model
                checkpoint_path = os.path.join(self.model_dir, f'checkpoint_user_{self.user_id}.keras')
                callbacks.append(ModelCheckpoint(
                    filepath=checkpoint_path,
                    save_best_only=True,
                    monitor='loss'
                ))

            # Adjust batch size based on dataset size
            adjusted_batch_size = min(batch_size, max(1, num_samples // 2))
            if adjusted_batch_size != batch_size:
                logger.info(f"Adjusted batch size from {batch_size} to {adjusted_batch_size} based on dataset size")

            # Execute training with error handling
            try:
                history = self.model.fit(
                    train_data[0], train_data[1],
                    validation_data=test_data,
                    epochs=epochs,
                    batch_size=adjusted_batch_size,
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=True
                ).history

                # Save the trained model
                self.save_model()

                return history

            except Exception as e:
                logger.error(f"Training failed with error: {str(e)}")
                logger.info("Attempting fallback training with simplified settings")

                # Create dummy recovery model if needed
                if self.model is None:
                    self.build_model()

                # Try minimal training to prevent complete failure
                history = self.model.fit(
                    train_data[0], train_data[1],
                    epochs=5,  # Minimal epochs
                    batch_size=1,  # Smallest possible batch
                    verbose=1
                ).history

                return history

        except Exception as e:
            logger.error(f"Critical error in training function: {str(e)}")
            logger.exception("Training exception details:")
            raise

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
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

            # Try saving to alternate location as backup
            try:
                backup_path = os.path.join(os.path.dirname(model_path),
                                           f"backup_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
                self.model.save(backup_path)
                logger.info(f"Saved backup model to {backup_path}")
                return True
            except:
                logger.error("Failed to save backup model")
                return False
