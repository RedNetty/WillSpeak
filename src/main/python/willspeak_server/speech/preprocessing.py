"""
Advanced audio preprocessing functions for WillSpeak, optimized for speech intelligibility
"""
import numpy as np
import librosa
import scipy.signal as signal
from loguru import logger

def preprocess_audio(audio_data, sample_rate, target_sr=16000, speech_focused=True):
    """
    Enhanced preprocessing pipeline for speech, focused on maintaining intelligibility
    and enhancing important speech features.

    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Original sample rate
        target_sr: Target sample rate
        speech_focused: Whether to apply speech-specific enhancements

    Returns:
        Preprocessed audio data
    """
    logger.info(f"Preprocessing audio: original sr={sample_rate}, target sr={target_sr}")

    # Guard against invalid inputs
    if audio_data is None or len(audio_data) == 0:
        logger.error("Invalid audio data provided")
        return np.zeros(1)

    # Convert to mono if necessary
    if len(audio_data.shape) > 1:
        logger.info("Converting stereo to mono")
        audio_data = librosa.to_mono(audio_data.T)

    # Resample if necessary
    if sample_rate != target_sr:
        logger.info(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)

    # Apply speech-specific enhancements if requested
    if speech_focused:
        # 1. Apply pre-emphasis filter to enhance higher frequencies (consonants)
        logger.info("Applying pre-emphasis filter for consonant clarity")
        audio_data = pre_emphasis_filter(audio_data, coefficient=0.95)

        # 2. Apply improved noise reduction
        logger.info("Applying speech-focused noise reduction")
        audio_data = reduce_noise(audio_data, target_sr)

        # 3. Enhance formants for vowel clarity
        logger.info("Enhancing formants for vowel intelligibility")
        audio_data = enhance_formants(audio_data, target_sr)

        # 4. Apply consonant enhancement
        logger.info("Enhancing consonants for articulation clarity")
        audio_data = enhance_consonants(audio_data, target_sr)

        # 5. Apply gentle dynamic range compression to improve intelligibility
        logger.info("Applying speech-optimized dynamic range compression")
        audio_data = compress_dynamic_range(audio_data, target_sr)
    else:
        # Apply basic normalization
        logger.info("Normalizing amplitude")
        audio_data = librosa.util.normalize(audio_data)

    # 6. Remove silence with improved speech-aware approach
    logger.info("Removing silence with speech-aware algorithm")
    audio_data = remove_silence_improved(audio_data, target_sr)

    # Final normalization to avoid clipping
    audio_data = librosa.util.normalize(audio_data) * 0.95

    logger.info("Preprocessing complete")
    return audio_data

def pre_emphasis_filter(audio_data, coefficient=0.95):
    """
    Apply pre-emphasis filter to enhance higher frequencies.
    This improves the clarity of consonants which are critical for intelligibility.

    Args:
        audio_data: Audio signal
        coefficient: Pre-emphasis coefficient (typically 0.95-0.97)

    Returns:
        Filtered audio signal
    """
    emphasized_audio = np.append(audio_data[0], audio_data[1:] - coefficient * audio_data[:-1])
    return emphasized_audio

def reduce_noise(audio_data, sample_rate, strength=0.5):
    """
    Spectral subtraction noise reduction optimized for preserving speech.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate
        strength: Noise reduction strength (0-1)

    Returns:
        Denoised audio signal
    """
    # Calculate STFT
    n_fft = 1024  # Smaller for better temporal resolution
    hop_length = 256
    stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)

    # Get magnitude and phase
    magnitude, phase = librosa.magphase(stft)

    # Estimate noise from the beginning and potential silence regions
    frame_energy = np.sum(np.abs(magnitude) ** 2, axis=0)
    sorted_energy = np.sort(frame_energy)
    noise_threshold = sorted_energy[int(len(sorted_energy) * 0.1)]  # Use lowest 10% energy frames
    noise_frames = frame_energy < noise_threshold

    # If we have enough noise frames, estimate noise profile
    if np.sum(noise_frames) > 3:
        noise_estimate = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
    else:
        # Fall back to using first few frames
        first_n = min(10, magnitude.shape[1])
        noise_estimate = np.mean(magnitude[:, :first_n], axis=1, keepdims=True)

    # Frequency-dependent noise reduction (less in speech range, more elsewhere)
    freq_bins = magnitude.shape[0]
    freq_weights = np.ones(freq_bins)

    # Define speech range (approximately 200-5000 Hz)
    speech_start = int(200 * freq_bins / (sample_rate/2))
    speech_end = int(5000 * freq_bins / (sample_rate/2))

    # Reduce noise more outside speech range
    freq_weights[:speech_start] = 1.2  # More reduction below speech range
    freq_weights[speech_start:speech_end] = 0.7  # Less reduction in speech range
    freq_weights[speech_end:] = 1.0  # Standard reduction above speech range

    # Apply weighted noise reduction
    scaled_noise = noise_estimate * strength * freq_weights[:, np.newaxis]
    magnitude_reduced = np.maximum(magnitude - scaled_noise, 0.05 * magnitude)

    # Apply temporal smoothing to avoid musical noise
    smoothed_magnitude = np.zeros_like(magnitude_reduced)
    for i in range(magnitude_reduced.shape[1]):
        if i < 2:
            smoothed_magnitude[:, i] = magnitude_reduced[:, i]
        else:
            # Simple 3-point smoothing
            smoothed_magnitude[:, i] = 0.2 * magnitude_reduced[:, i-2] + \
                                      0.3 * magnitude_reduced[:, i-1] + \
                                      0.5 * magnitude_reduced[:, i]

    # Reconstruct signal
    stft_reduced = smoothed_magnitude * phase
    denoised_audio = librosa.istft(stft_reduced, hop_length=hop_length, n_fft=n_fft, length=len(audio_data))

    return denoised_audio

def enhance_formants(audio_data, sample_rate):
    """
    Enhance vowel formants for improved clarity.
    Uses spectral peak enhancement in typical formant frequency regions.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate

    Returns:
        Formant-enhanced audio
    """
    # Calculate STFT
    n_fft = 2048  # Larger FFT for better frequency resolution
    hop_length = 512
    stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)

    # Get magnitude and phase
    magnitude, phase = librosa.magphase(stft)

    # Define formant regions (approximate ranges in Hz)
    freq_bins = magnitude.shape[0]
    nyquist = sample_rate / 2

    # Approximate formant ranges
    f1_range = (300, 1000)    # First formant (F1)
    f2_range = (1000, 2500)   # Second formant (F2)
    f3_range = (2500, 4000)   # Third formant (F3)

    # Convert to bin indices
    f1_bins = (int(f1_range[0] * freq_bins / nyquist), int(f1_range[1] * freq_bins / nyquist))
    f2_bins = (int(f2_range[0] * freq_bins / nyquist), int(f2_range[1] * freq_bins / nyquist))
    f3_bins = (int(f3_range[0] * freq_bins / nyquist), int(f3_range[1] * freq_bins / nyquist))

    # Create formant enhancement filter
    formant_filter = np.ones(freq_bins)

    # Enhance formant regions with different emphasis factors
    formant_filter[f1_bins[0]:f1_bins[1]] = 1.7  # Stronger emphasis on F1 (important for vowel identity)
    formant_filter[f2_bins[0]:f2_bins[1]] = 1.5  # Medium emphasis on F2
    formant_filter[f3_bins[0]:f3_bins[1]] = 1.3  # Lighter emphasis on F3

    # Apply filter to enhance formants
    enhanced_magnitude = magnitude * formant_filter[:, np.newaxis]

    # Smooth the enhanced spectrum to avoid artifacts
    for i in range(enhanced_magnitude.shape[1]):
        for j in range(1, enhanced_magnitude.shape[0]-1):
            enhanced_magnitude[j, i] = 0.25 * enhanced_magnitude[j-1, i] + \
                                      0.5 * enhanced_magnitude[j, i] + \
                                      0.25 * enhanced_magnitude[j+1, i]

    # Reconstruct signal
    enhanced_stft = enhanced_magnitude * phase
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, n_fft=n_fft, length=len(audio_data))

    return enhanced_audio

def enhance_consonants(self, audio_data):
    """
    Enhanced consonant detection and boost for clearer articulation.
    Focuses on high-frequency components and transients.

    Args:
        audio_data: Input audio signal

    Returns:
        Consonant-enhanced audio signal
    """
    # First detect potential consonant regions using energy and zero-crossing rate
    frame_length = 256
    hop_length = 64

    # Calculate frame-wise energy and zero-crossing rate
    energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=frame_length, hop_length=hop_length)[0]

    # Normalize values
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
    zcr = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-8)

    # Consonants typically have high ZCR and either high or low energy
    consonant_score = zcr * 2 - energy * 0.5  # Emphasize ZCR more than energy

    # Normalize the score
    consonant_score = (consonant_score - np.min(consonant_score)) / (np.max(consonant_score) - np.min(consonant_score) + 1e-8)

    # Convert frame-wise score to sample-wise mask
    sample_consonant_score = np.repeat(consonant_score, hop_length)
    # Pad if necessary
    if len(sample_consonant_score) < len(audio_data):
        sample_consonant_score = np.pad(sample_consonant_score,
                                       (0, len(audio_data) - len(sample_consonant_score)))
    elif len(sample_consonant_score) > len(audio_data):
        sample_consonant_score = sample_consonant_score[:len(audio_data)]

    # Apply high-shelf filter to boost high frequencies (where consonants are)
    # First get the STFTs
    stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
    magnitude, phase = librosa.magphase(stft)

    # Create a high-shelf filter (boost high frequencies)
    shelf_freq = 3000  # Hz
    bin_shelf = int(shelf_freq * self.n_fft / self.sr)
    high_shelf = np.ones(magnitude.shape[0])
    high_shelf[bin_shelf:] = self.high_freq_gain  # Boost high frequencies

    # Apply the filter with time-varying strength based on consonant score
    consonant_frames = librosa.util.fix_length(sample_consonant_score, size=magnitude.shape[1] * self.hop_length)
    consonant_frames = librosa.util.frame(consonant_frames, frame_length=self.hop_length, hop_length=self.hop_length)
    consonant_frames = np.mean(consonant_frames, axis=0)

    # Create time-varying filter strength
    filter_strength = np.ones(magnitude.shape[1])
    for i in range(len(filter_strength)):
        # Scale between 1.0 and consonant_emphasis based on consonant score
        filter_strength[i] = 1.0 + consonant_frames[i] * (self.consonant_emphasis - 1.0)

    # Apply the filter
    enhanced_magnitude = magnitude.copy()
    for i in range(magnitude.shape[1]):
        enhanced_magnitude[:, i] = magnitude[:, i] * (1 + (high_shelf - 1) * filter_strength[i])

    # Reconstruct the signal
    enhanced_stft = enhanced_magnitude * phase
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length, win_length=self.n_fft)

    # Ensure the output length matches the input length
    if len(enhanced_audio) > len(audio_data):
        enhanced_audio = enhanced_audio[:len(audio_data)]
    elif len(enhanced_audio) < len(audio_data):
        enhanced_audio = np.pad(enhanced_audio, (0, len(audio_data) - len(enhanced_audio)))

    return enhanced_audio

def compress_dynamic_range(audio_data, sample_rate, threshold=-24, ratio=2.5, attack=0.005, release=0.05):
    """
    Apply dynamic range compression optimized for speech intelligibility.
    Makes quieter sounds (like consonants) more audible relative to louder sounds.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate
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

    # Apply frequency-dependent envelope follower with optimized time constants
    # This maintains consonant attacks while smoothing vowels

    # Attack and release in samples
    attack_samples = int(attack * sample_rate)
    if attack_samples < 1:
        attack_samples = 1
    release_samples = int(release * sample_rate)
    if release_samples < 1:
        release_samples = 1

    # Attack and release coefficients
    attack_coef = np.exp(-1.0 / attack_samples)
    release_coef = np.exp(-1.0 / release_samples)

    # Initialize envelope tracker
    envelope = np.zeros_like(audio_data)

    # First-pass envelope detection
    for i in range(1, len(audio_data)):
        if abs_audio[i] > envelope[i - 1]:
            # Attack phase
            envelope[i] = attack_coef * envelope[i - 1] + (1 - attack_coef) * abs_audio[i]
        else:
            # Release phase
            envelope[i] = release_coef * envelope[i - 1] + (1 - release_coef) * abs_audio[i]

    # Calculate gain
    gain = np.ones_like(audio_data)
    mask = envelope > threshold_linear
    gain[mask] = (envelope[mask] / threshold_linear) ** (1 / ratio - 1)

    # Smooth gain changes to avoid artifacts
    win_length = int(0.01 * sample_rate)  # 10ms window
    if win_length % 2 == 0:
        win_length += 1  # Make sure window length is odd

    # Apply smoothing filter to gain
    gain = signal.savgol_filter(gain, win_length, 2)

    # Apply gain
    compressed_audio = audio_data * gain

    # Normalize to avoid clipping
    if np.max(np.abs(compressed_audio)) > 0.99:
        compressed_audio = compressed_audio / np.max(np.abs(compressed_audio)) * 0.99

    return compressed_audio

def remove_silence_improved(audio_data, sample_rate, top_db=20, min_silence_duration=0.1, min_speech_duration=0.05):
    """
    Remove silence with improved parameters for speech and ensure continuous speech segments.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate
        top_db: Threshold for considering a frame as silence
        min_silence_duration: Minimum silence duration in seconds to remove
        min_speech_duration: Minimum speech segment duration to keep

    Returns:
        Audio with silence removed
    """
    # Calculate frame length and hop length based on minimum durations
    frame_length = 1024
    hop_length = 256

    min_silence_frames = int(min_silence_duration * sample_rate / hop_length)
    min_speech_frames = int(min_speech_duration * sample_rate / hop_length)

    # Detect non-silent frames
    intervals = librosa.effects.split(
        audio_data,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Filter intervals for minimum duration
    filtered_intervals = []
    for start, end in intervals:
        duration_frames = (end - start) // hop_length
        if duration_frames >= min_speech_frames:
            filtered_intervals.append((start, end))

    # If we filtered all intervals, return a short segment of the original
    if not filtered_intervals:
        logger.warning("All speech segments were too short, returning partial audio")
        mid_point = len(audio_data) // 2
        segment_size = int(0.5 * sample_rate)  # 0.5 seconds
        start = max(0, mid_point - segment_size // 2)
        end = min(len(audio_data), mid_point + segment_size // 2)
        return audio_data[start:end]

    # Merge intervals that are close together
    merged_intervals = [filtered_intervals[0]]
    for interval in filtered_intervals[1:]:
        current_start, current_end = interval
        prev_start, prev_end = merged_intervals[-1]

        # If current interval is close to previous, merge them
        if current_start - prev_end < min_silence_frames * hop_length:
            merged_intervals[-1] = (prev_start, current_end)
        else:
            merged_intervals.append(interval)

    # Concatenate non-silent segments
    non_silent_audio = []
    for start, end in merged_intervals:
        non_silent_audio.append(audio_data[start:end])

    # Add small crossfades between segments if we have multiple segments
    if len(non_silent_audio) > 1:
        crossfade_length = int(0.02 * sample_rate)  # 20ms crossfade

        result = [non_silent_audio[0]]
        for segment in non_silent_audio[1:]:
            # Create crossfade for smooth transition
            if len(result[-1]) > crossfade_length and len(segment) > crossfade_length:
                fade_out = result[-1][-crossfade_length:] * np.linspace(1, 0, crossfade_length)
                fade_in = segment[:crossfade_length] * np.linspace(0, 1, crossfade_length)
                crossfade = fade_out + fade_in

                # Combine segments with crossfade
                result[-1] = result[-1][:-crossfade_length]
                result.append(crossfade)
                result.append(segment[crossfade_length:])
            else:
                result.append(segment)

        # Concatenate the result
        audio_without_silence = np.concatenate(result)
    else:
        audio_without_silence = np.concatenate(non_silent_audio)

    return audio_without_silence

def extract_speech_features(audio_data, sample_rate):
    """
    Extract speech-specific features for analysis and enhancement.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate

    Returns:
        Dictionary of speech features
    """
    # Initialize features dictionary
    features = {}

    # 1. Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio_data,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=1024,
        hop_length=256
    )
    features['mfcc'] = mfcc
    features['mfcc_mean'] = np.mean(mfcc, axis=1)
    features['mfcc_std'] = np.std(mfcc, axis=1)

    # 2. Extract pitch (F0) information
    f0, voiced_flag, _ = librosa.pyin(
        audio_data,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate
    )

    # Replace NaN with zeros
    f0 = np.nan_to_num(f0)

    # Calculate pitch statistics for voiced regions
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        features['pitch_mean'] = np.mean(voiced_f0)
        features['pitch_std'] = np.std(voiced_f0)
        features['pitch_min'] = np.min(voiced_f0)
        features['pitch_max'] = np.max(voiced_f0)
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['pitch_min'] = 0
        features['pitch_max'] = 0

    features['voiced_ratio'] = np.mean(voiced_flag)

    # 3. Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio_data,
        sr=sample_rate
    )[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data,
        sr=sample_rate
    )[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio_data,
        sr=sample_rate
    )[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

    # 4. Extract consonant-specific metrics
    # Zero crossing rate is higher for consonants
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # 5. Extract rhythm and tempo features
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
    features['tempo'] = tempo

    # 6. Extract energy features
    rms = librosa.feature.rms(y=audio_data)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)

    # 7. Signal-to-noise ratio estimate
    signal_power = np.mean(audio_data**2)

    # Estimate noise from silent or low-energy regions
    sorted_rms = np.sort(rms)
    noise_threshold = sorted_rms[int(len(sorted_rms) * 0.1)]  # Use lowest 10% energy frames
    noise_frames = rms < noise_threshold

    if np.any(noise_frames):
        # Calculate frame indices
        hop_length = 512  # Default for RMS calculation
        noise_indices = np.where(noise_frames)[0]

        # Create a mask for noise samples
        noise_mask = np.zeros_like(audio_data, dtype=bool)
        for frame in noise_indices:
            start = frame * hop_length
            end = min(start + hop_length, len(noise_mask))
            noise_mask[start:end] = True

        # Calculate noise power
        if np.any(noise_mask):
            noise_power = np.mean(audio_data[noise_mask]**2)

            # Calculate SNR
            if noise_power > 0:
                features['snr_estimate'] = 10 * np.log10(signal_power / noise_power)
            else:
                features['snr_estimate'] = 100  # High SNR if no noise detected
        else:
            features['snr_estimate'] = 30  # Default value
    else:
        features['snr_estimate'] = 30  # Default value

    return features