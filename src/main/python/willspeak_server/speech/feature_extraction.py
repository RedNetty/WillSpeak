"""
Enhanced audio feature extraction for WillSpeak with speech-specific improvements
"""
import numpy as np
import librosa
from python_speech_features import mfcc
from scipy import signal
from loguru import logger
from scipy import signal

def extract_features(audio_data, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512, speech_focused=True):
    """
    Extract features from audio data with enhanced focus on speech-specific characteristics:
    - MFCC features with speech-specific configurations
    - Articulation features for consonant clarity measurement
    - Formant structure analysis for vowel quality assessment
    - Prosodic features (rhythm, intonation)
    - Speech energy distribution and spectral balance

    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for feature extraction
        speech_focused: Whether to include speech-specific advanced features

    Returns:
        Dictionary of features
    """
    logger.info(f"Extracting speech-focused features: n_mfcc={n_mfcc}, n_fft={n_fft}")

    # Initialize features dictionary
    features = {}

    # 1. Basic spectral features
    # Extract MFCC with speech optimizations
    mfcc_features = mfcc(
        audio_data,
        samplerate=sample_rate,
        numcep=n_mfcc,
        nfilt=26,  # More filters for better frequency resolution
        nfft=n_fft,
        preemph=0.97,  # Higher pre-emphasis for consonant clarity
        ceplifter=22,  # Standard liftering
        appendEnergy=True  # Include energy
    )

    # Calculate mean and std of MFCCs and their deltas
    mfcc_mean = np.mean(mfcc_features, axis=0)
    mfcc_std = np.std(mfcc_features, axis=0)

    # Calculate delta and delta-delta coefficients for speech dynamics
    mfcc_delta = librosa.feature.delta(mfcc_features)
    mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)

    mfcc_delta_mean = np.mean(mfcc_delta, axis=0)
    mfcc_delta_std = np.std(mfcc_delta, axis=0)
    mfcc_delta2_mean = np.mean(mfcc_delta2, axis=0)
    mfcc_delta2_std = np.std(mfcc_delta2, axis=0)

    # Add basic MFCC features to dictionary
    features["mfcc_features"] = mfcc_features
    features["mfcc_mean"] = mfcc_mean
    features["mfcc_std"] = mfcc_std
    features["mfcc_delta_mean"] = mfcc_delta_mean
    features["mfcc_delta_std"] = mfcc_delta_std
    features["mfcc_delta2_mean"] = mfcc_delta2_mean
    features["mfcc_delta2_std"] = mfcc_delta2_std

    # 2. Extract standard spectral features
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

    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    # Add spectral features to dictionary
    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    features["spectral_centroid_std"] = float(np.std(spectral_centroid))
    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
    features["spectral_contrast_mean"] = [float(np.mean(band)) for band in spectral_contrast]
    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

    # If speech-focused analysis is requested, add more specialized features
    if speech_focused:
        # 3. Extract formant structure features
        formant_features = extract_formant_features(audio_data, sample_rate)
        features.update(formant_features)

        # 4. Extract articulation features (consonant clarity)
        articulation_features = extract_articulation_features(audio_data, sample_rate)
        features.update(articulation_features)

        # 5. Extract prosodic features (rhythm, intonation)
        prosodic_features = extract_prosodic_features(audio_data, sample_rate)
        features.update(prosodic_features)

        # 6. Extract voice quality features
        voice_quality_features = extract_voice_quality_features(audio_data, sample_rate)
        features.update(voice_quality_features)

        # 7. Extract speech clarity metrics
        clarity_features = extract_clarity_metrics(audio_data, sample_rate)
        features.update(clarity_features)

    logger.info("Feature extraction complete")
    return features


def extract_formant_features(audio_data, sample_rate):
    """
    Extract formant structure features for vowel quality assessment.
    Uses LPC analysis to estimate formant frequencies and bandwidths.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate

    Returns:
        Dictionary of formant features
    """
    features = {}

    # Frame the audio into 25ms frames with 10ms step
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)

    # Pre-emphasis to enhance formant structure
    preemph_frames = np.zeros_like(frames)
    for i in range(frames.shape[1]):
        preemph_frames[:, i] = np.append(frames[0, i], frames[1:, i] - 0.97 * frames[:-1, i])

    # Apply a window function
    window = np.hamming(frame_length)
    windowed_frames = preemph_frames * window[:, np.newaxis]

    # LPC analysis for formant estimation
    num_formants = 5  # Number of formants to estimate (F1-F5)
    lpc_order = 2 + num_formants * 2  # Rule of thumb: 2 + 2 * number of formants

    formant_freqs = []
    formant_bandwidths = []

    for i in range(windowed_frames.shape[1]):
        frame = windowed_frames[:, i]

        # Calculate LPC coefficients
        lpc_coeffs = librosa.lpc(frame, order=lpc_order)

        # Get roots of the LPC polynomial
        roots = np.roots(lpc_coeffs)

        # Keep only roots with positive imaginary part and inside unit circle
        roots = roots[np.abs(roots) < 0.99]
        roots = roots[np.imag(roots) > 0]

        # Calculate frequencies and bandwidths
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * sample_rate / (2 * np.pi)

        # Calculate bandwidths using the formula:
        # bandwidth = -ln(|r|) * sample_rate / π
        bw = -1 * np.log(np.abs(roots)) * sample_rate / np.pi

        # Sort by frequency
        sorted_idx = np.argsort(freqs)
        sorted_freqs = freqs[sorted_idx]
        sorted_bw = bw[sorted_idx]

        # Store formant data from this frame
        if len(sorted_freqs) >= num_formants:
            formant_freqs.append(sorted_freqs[:num_formants])
            formant_bandwidths.append(sorted_bw[:num_formants])

    # Calculate statistics if we have enough data
    if formant_freqs:
        formant_freqs_array = np.array(formant_freqs)
        formant_bw_array = np.array(formant_bandwidths)

        # Mean formant frequencies
        for i in range(num_formants):
            if i < formant_freqs_array.shape[1]:
                features[f"F{i + 1}_mean"] = float(np.mean(formant_freqs_array[:, i]))
                features[f"F{i + 1}_std"] = float(np.std(formant_freqs_array[:, i]))
                features[f"F{i + 1}_bw_mean"] = float(np.mean(formant_bw_array[:, i]))

        # Calculate formant ratios (important for vowel quality)
        if formant_freqs_array.shape[1] >= 2:
            features["F2_F1_ratio"] = float(np.mean(formant_freqs_array[:, 1] / formant_freqs_array[:, 0]))

        if formant_freqs_array.shape[1] >= 3:
            features["F3_F2_ratio"] = float(np.mean(formant_freqs_array[:, 2] / formant_freqs_array[:, 1]))

        # Formant dispersion (average distance between adjacent formants)
        if formant_freqs_array.shape[1] >= 3:
            dispersions = []
            for i in range(formant_freqs_array.shape[0]):
                frame_dispersions = np.diff(formant_freqs_array[i, :3])
                dispersions.extend(frame_dispersions)

            if dispersions:
                features["formant_dispersion"] = float(np.mean(dispersions))
    else:
        # Default values if formant analysis fails
        for i in range(num_formants):
            features[f"F{i + 1}_mean"] = 0
            features[f"F{i + 1}_std"] = 0
            features[f"F{i + 1}_bw_mean"] = 0

        features["F2_F1_ratio"] = 0
        features["F3_F2_ratio"] = 0
        features["formant_dispersion"] = 0

    return features


def extract_articulation_features(audio_data, sample_rate):
    """
    Extract articulation features for consonant clarity measurement.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate

    Returns:
        Dictionary of articulation features
    """
    features = {}

    # 1. Zero-crossing rate (higher for fricatives and other consonants)
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))
    features["zcr_max"] = float(np.max(zcr))

    # 2. High frequency energy (important for fricatives)
    # Apply high-pass filter
    nyquist = sample_rate / 2
    cutoff = 3000  # Hz - focus on frequencies important for consonants

    b, a = signal.butter(4, cutoff / nyquist, 'highpass')
    high_passed = signal.filtfilt(b, a, audio_data)

    # Calculate energy in high-frequency band
    high_freq_energy = np.sum(high_passed ** 2) / np.sum(audio_data ** 2)
    features["high_freq_energy_ratio"] = float(high_freq_energy)

    # 3. Spectral flux (rate of change in spectrum, higher for consonant transitions)
    hop_length = int(0.010 * sample_rate)  # 10ms hop
    stft = np.abs(librosa.stft(audio_data, hop_length=hop_length))

    # Normalize each frame
    stft_norm = librosa.util.normalize(stft, axis=0)

    # Calculate frame-to-frame spectral difference
    spectral_diff = np.zeros(stft_norm.shape[1] - 1)
    for i in range(len(spectral_diff)):
        spectral_diff[i] = np.sum((stft_norm[:, i + 1] - stft_norm[:, i]) ** 2)

    features["spectral_flux_mean"] = float(np.mean(spectral_diff))
    features["spectral_flux_std"] = float(np.std(spectral_diff))

    # 4. Speech rate estimate using onsets
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate)

    # Estimate speech rate from onset density
    if len(audio_data) > 0:
        duration = len(audio_data) / sample_rate
        if duration > 0:
            speech_rate = len(onsets) / duration
            features["speech_rate"] = float(speech_rate)
        else:
            features["speech_rate"] = 0.0
    else:
        features["speech_rate"] = 0.0

    # 5. Envelope modulation (important for syllable structure)
    # Get speech envelope
    envelope = np.abs(librosa.hilbert(audio_data))

    # Low-pass filter to get syllable-level modulations
    b, a = signal.butter(2, 8 / nyquist, 'lowpass')
    envelope_filtered = signal.filtfilt(b, a, envelope)

    # Calculate envelope modulation metrics
    if len(envelope_filtered) > 0:
        features["envelope_modulation_depth"] = float(np.max(envelope_filtered) / np.mean(envelope_filtered))
        features["envelope_variance"] = float(np.var(envelope_filtered))
    else:
        features["envelope_modulation_depth"] = 0.0
        features["envelope_variance"] = 0.0

    return features


def extract_prosodic_features(audio_data, sample_rate):
    """
    Extract prosodic features (rhythm, intonation).

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate

    Returns:
        Dictionary of prosodic features
    """
    features = {}

    # 1. Pitch (F0) estimation and statistics
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=sample_rate,
            fill_na=None
        )

        # Filter out NaN values (unvoiced regions)
        f0_no_nan = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        if len(f0_no_nan) > 0:
            features["f0_mean"] = float(np.mean(f0_no_nan))
            features["f0_std"] = float(np.std(f0_no_nan))
            features["f0_min"] = float(np.min(f0_no_nan))
            features["f0_max"] = float(np.max(f0_no_nan))
            features["f0_range"] = float(np.max(f0_no_nan) - np.min(f0_no_nan))

            # Calculate pitch dynamism (variance normalized by mean)
            if features["f0_mean"] > 0:
                features["f0_dynamism"] = float(features["f0_std"] / features["f0_mean"])
            else:
                features["f0_dynamism"] = 0.0

            # Voiced-to-unvoiced ratio
            features["voiced_ratio"] = float(np.mean(~np.isnan(f0)))
        else:
            # Default values if pitch estimation fails
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0
            features["f0_min"] = 0.0
            features["f0_max"] = 0.0
            features["f0_range"] = 0.0
            features["f0_dynamism"] = 0.0
            features["voiced_ratio"] = 0.0
    except Exception as e:
        logger.warning(f"Pitch estimation failed: {e}")
        # Default values
        features["f0_mean"] = 0.0
        features["f0_std"] = 0.0
        features["f0_min"] = 0.0
        features["f0_max"] = 0.0
        features["f0_range"] = 0.0
        features["f0_dynamism"] = 0.0
        features["voiced_ratio"] = 0.0

    # 2. Energy contour analysis
    rms = librosa.feature.rms(y=audio_data)[0]
    if len(rms) > 0:
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))
        features["energy_range"] = float(np.max(rms) - np.min(rms))

        # Energy dynamism
        if features["energy_mean"] > 0:
            features["energy_dynamism"] = float(features["energy_std"] / features["energy_mean"])
        else:
            features["energy_dynamism"] = 0.0
    else:
        features["energy_mean"] = 0.0
        features["energy_std"] = 0.0
        features["energy_range"] = 0.0
        features["energy_dynamism"] = 0.0

    # 3. Speech rhythm metrics
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    features["tempo_estimate"] = float(tempo)

    # 4. Syllable rate estimate
    # Using the envelope to estimate syllables
    envelope = np.abs(librosa.hilbert(audio_data))

    # Low-pass filter to get syllable-level modulations
    nyquist = sample_rate / 2
    b, a = signal.butter(2, 8 / nyquist, 'lowpass')
    envelope_filtered = signal.filtfilt(b, a, envelope)

    # Find peaks in the envelope as syllable nuclei
    peaks, _ = signal.find_peaks(envelope_filtered, distance=int(0.15 * sample_rate))

    # Calculate syllable rate
    if len(audio_data) > 0:
        duration = len(audio_data) / sample_rate
        if duration > 0:
            syllable_rate = len(peaks) / duration
            features["syllable_rate"] = float(syllable_rate)
        else:
            features["syllable_rate"] = 0.0
    else:
        features["syllable_rate"] = 0.0

    return features


def extract_voice_quality_features(audio_data, sample_rate):
    """
    Extract voice quality features (breathiness, creakiness, strain).

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate

    Returns:
        Dictionary of voice quality features
    """
    features = {}

    # 1. Harmonics-to-noise ratio (HNR) - higher for clearer voice
    try:
        # We can estimate HNR using autocorrelation method
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)

        hnr_values = []
        for i in range(frames.shape[1]):
            frame = frames[:, i] * np.hamming(frame_length)
            if np.sum(frame ** 2) > 0:  # Ensure frame has energy
                # Calculate autocorrelation
                autocorr = librosa.autocorrelate(frame)

                # Find first peak after the zero lag
                peaks, _ = signal.find_peaks(autocorr[1:], height=0)
                if len(peaks) > 0:
                    peak_idx = peaks[0] + 1  # Adjust for the slicing

                    if autocorr[0] > 0:  # Avoid division by zero
                        # Calculate HNR using autocorrelation ratio
                        hnr = autocorr[peak_idx] / (autocorr[0] - autocorr[peak_idx])
                        if hnr > 0:
                            hnr_db = 10 * np.log10(hnr)
                            hnr_values.append(hnr_db)

        if hnr_values:
            features["hnr_mean"] = float(np.mean(hnr_values))
            features["hnr_std"] = float(np.std(hnr_values))
        else:
            features["hnr_mean"] = 0.0
            features["hnr_std"] = 0.0
    except Exception as e:
        logger.warning(f"HNR calculation failed: {e}")
        features["hnr_mean"] = 0.0
        features["hnr_std"] = 0.0

    # 2. Jitter (frequency perturbation)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )

        # Calculate jitter only on voiced regions
        if f0 is not None:
            f0_voiced = f0[~np.isnan(f0)]

            if len(f0_voiced) > 1:
                # Local jitter: mean absolute difference between consecutive periods
                period_diff = np.abs(np.diff(1.0 / f0_voiced))
                local_jitter = np.mean(period_diff) / np.mean(1.0 / f0_voiced)
                features["jitter_local"] = float(local_jitter)

                # PPQ5 (five-point period perturbation quotient)
                if len(f0_voiced) >= 5:
                    periods = 1.0 / f0_voiced
                    ppq5_values = []
                    for i in range(2, len(periods) - 2):
                        avg_period = np.mean(periods[i - 2:i + 3])
                        if avg_period > 0:
                            ppq5 = np.abs(periods[i] - avg_period) / avg_period
                            ppq5_values.append(ppq5)

                    if ppq5_values:
                        features["jitter_ppq5"] = float(np.mean(ppq5_values))
                    else:
                        features["jitter_ppq5"] = 0.0
                else:
                    features["jitter_ppq5"] = 0.0
            else:
                features["jitter_local"] = 0.0
                features["jitter_ppq5"] = 0.0
        else:
            features["jitter_local"] = 0.0
            features["jitter_ppq5"] = 0.0
    except Exception as e:
        logger.warning(f"Jitter calculation failed: {e}")
        features["jitter_local"] = 0.0
        features["jitter_ppq5"] = 0.0

    # 3. Shimmer (amplitude perturbation)
    try:
        # Get amplitude envelope
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)

        # RMS energy per frame
        rms_frames = librosa.feature.rms(
            y=audio_data,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        if len(rms_frames) > 1:
            # Local shimmer: mean absolute difference between consecutive amplitudes
            amp_diff = np.abs(np.diff(rms_frames))
            if np.mean(rms_frames[:-1]) > 0:
                local_shimmer = np.mean(amp_diff) / np.mean(rms_frames[:-1])
                features["shimmer_local"] = float(local_shimmer)
            else:
                features["shimmer_local"] = 0.0

            # APQ5 (five-point amplitude perturbation quotient)
            if len(rms_frames) >= 5:
                apq5_values = []
                for i in range(2, len(rms_frames) - 2):
                    avg_amp = np.mean(rms_frames[i - 2:i + 3])
                    if avg_amp > 0:
                        apq5 = np.abs(rms_frames[i] - avg_amp) / avg_amp
                        apq5_values.append(apq5)

                if apq5_values:
                    features["shimmer_apq5"] = float(np.mean(apq5_values))
                else:
                    features["shimmer_apq5"] = 0.0
            else:
                features["shimmer_apq5"] = 0.0
        else:
            features["shimmer_local"] = 0.0
            features["shimmer_apq5"] = 0.0
    except Exception as e:
        logger.warning(f"Shimmer calculation failed: {e}")
        features["shimmer_local"] = 0.0
        features["shimmer_apq5"] = 0.0

    # 4. Cepstral Peak Prominence (CPP) - smoothness of spectrum
    try:
        # Calculate spectrum
        spectrum = np.abs(librosa.stft(audio_data))

        # Calculate log power spectrum
        log_power = librosa.amplitude_to_db(spectrum)

        # Calculate CPP for each frame
        cpp_values = []
        for i in range(log_power.shape[1]):
            frame_spectrum = log_power[:, i]

            # Get cepstrum (inverse DFT of log spectrum)
            cepstrum = np.fft.ifft(frame_spectrum).real

            # Find the cepstral peak in the expected quefrency range for pitch
            # (usually between 2ms and 20ms, or 50-500 Hz)
            min_idx = int(sample_rate / 500 / (sample_rate / log_power.shape[0]))
            max_idx = int(sample_rate / 50 / (sample_rate / log_power.shape[0]))

            if min_idx < max_idx and max_idx < len(cepstrum):
                peak_idx = np.argmax(cepstrum[min_idx:max_idx]) + min_idx

                # Calculate regression line
                x = np.arange(len(cepstrum))
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, cepstrum, rcond=None)[0]
                regression_line = m * x + c

                # CPP is the difference between the cepstral peak and the regression line at that point
                cpp = cepstrum[peak_idx] - regression_line[peak_idx]
                cpp_values.append(cpp)

        if cpp_values:
            features["cpp_mean"] = float(np.mean(cpp_values))
            features["cpp_std"] = float(np.std(cpp_values))
        else:
            features["cpp_mean"] = 0.0
            features["cpp_std"] = 0.0
    except Exception as e:
        logger.warning(f"CPP calculation failed: {e}")
        features["cpp_mean"] = 0.0
        features["cpp_std"] = 0.0

    return features


def extract_clarity_metrics(audio_data, sample_rate):
    """
    Extract overall speech clarity metrics.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate

    Returns:
        Dictionary of clarity features
    """
    features = {}

    # 1. Speech-to-noise ratio estimation
    try:
        # Get signal power
        signal_power = np.mean(audio_data ** 2)

        # Estimate noise from low-energy segments
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)

        rms = librosa.feature.rms(
            y=audio_data,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Sort RMS values and take lowest 10% as noise estimate
        sorted_rms = np.sort(rms)
        noise_threshold = sorted_rms[int(len(sorted_rms) * 0.1)]

        noise_frames = np.where(rms <= noise_threshold)[0]

        if len(noise_frames) > 0:
            # Reconstruct noise signal
            noise_signal = np.zeros_like(audio_data)
            for frame in noise_frames:
                start = frame * hop_length
                end = start + frame_length
                if end <= len(noise_signal):
                    noise_signal[start:end] = audio_data[start:end]

            # Calculate noise power
            noise_power = np.mean(noise_signal ** 2)

            # Calculate SNR
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                features["snr_estimate"] = float(snr)
            else:
                features["snr_estimate"] = 30.0  # High SNR if no noise detected
        else:
            features["snr_estimate"] = 30.0  # High SNR if no noise detected
    except Exception as e:
        logger.warning(f"SNR estimation failed: {e}")
        features["snr_estimate"] = 0.0

    # 5. Overall articulation rate (syllables per second)
    try:
        # Using the envelope to estimate syllables
        # Use scipy.signal.hilbert instead of librosa.hilbert
        envelope = np.abs(signal.hilbert(audio_data))

        # Low-pass filter to get syllable-level modulations
        nyquist = sample_rate / 2
        b, a = signal.butter(2, 8 / nyquist, 'lowpass')
        envelope_filtered = signal.filtfilt(b, a, envelope)

        # Find peaks in the envelope as syllable nuclei
        peaks, _ = signal.find_peaks(envelope_filtered, distance=int(0.15 * sample_rate))

        # Calculate syllable rate
        if len(audio_data) > 0:
            duration = len(audio_data) / sample_rate
            if duration > 0:
                syllable_rate = len(peaks) / duration
                features["syllable_rate"] = float(syllable_rate)
            else:
                features["syllable_rate"] = 0.0
        else:
            features["syllable_rate"] = 0.0
    except Exception as e:
        logger.warning(f"Syllable rate calculation failed: {e}")
        features["syllable_rate"] = 0.0

    return features


def extract_speech_comprehensibility_score(self, audio_data, sample_rate):
    """
    Calculate an overall score for speech comprehensibility.
    This is a composite score based on multiple features.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate

    Returns:
        Float: Comprehensibility score from 0-1
    """
    # Extract all relevant features
    features = extract_features(audio_data, sample_rate, speech_focused=True)

    # Define feature weights for overall comprehensibility
    weights = {
        # Formant structure (vowel clarity)
        "F1_mean": 0.05,
        "F2_F1_ratio": 0.10,
        "formant_dispersion": 0.05,

        # Articulation features
        "high_freq_energy_ratio": 0.15,  # Consonant clarity
        "spectral_flux_mean": 0.05,

        # Voice quality
        "hnr_mean": 0.10,
        "jitter_local": -0.05,  # Negative weight (lower is better)
        "shimmer_local": -0.05,  # Negative weight (lower is better)

        # Speech rhythm
        "syllable_rate": 0.05,

        # Overall clarity
        "snr_estimate": 0.15,
        "articulation_index": 0.15,
        "vowel_space_area": 0.10
    }

    # Normalize each feature to 0-1 range based on typical values
    normalizers = {
        "F1_mean": lambda x: min(max(x / 600, 0), 1),
        "F2_F1_ratio": lambda x: min(max((x - 3) / 3, 0), 1),
        "formant_dispersion": lambda x: min(max(x / 1000, 0), 1),
        "high_freq_energy_ratio": lambda x: min(max(x * 5, 0), 1),
        "spectral_flux_mean": lambda x: min(max(x * 10, 0), 1),
        "hnr_mean": lambda x: min(max((x + 10) / 30, 0), 1),  # Typically -10 to 20 dB
        "jitter_local": lambda x: 1 - min(max(x * 20, 0), 1),  # Invert so lower is better
        "shimmer_local": lambda x: 1 - min(max(x * 10, 0), 1),  # Invert so lower is better
        "syllable_rate": lambda x: min(max(x / 5, 0), 1),  # Typical range 0-5 syllables/sec
        "snr_estimate": lambda x: min(max(x / 30, 0), 1),  # 0-30 dB
        "articulation_index": lambda x: x,  # Already 0-1
        "vowel_space_area": lambda x: min(max(x / 250000, 0), 1)
    }

    # Calculate weighted score
    score = 0
    total_weight = 0

    for feature, weight in weights.items():
        if feature in features and features[feature] is not None:
            # Apply normalization if available
            if feature in normalizers:
                normalized_value = normalizers[feature](features[feature])
            else:
                # Default normalization (assume 0-1 range)
                normalized_value = max(min(features[feature], 1), 0)

            # Add to weighted score
            score += weight * normalized_value
            total_weight += abs(weight)

    # Normalize final score to 0-1 range
    if total_weight > 0:
        final_score = score / total_weight
    else:
        final_score = 0.5  # Default middle value if we couldn't calculate

    # Ensure score is in 0-1 range
    return max(min(final_score, 1), 0)
