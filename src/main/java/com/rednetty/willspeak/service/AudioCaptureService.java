package com.rednetty.willspeak.service;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import javax.sound.sampled.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Service for capturing audio from microphone input and managing audio data.
 */
public class AudioCaptureService {
    private static final Logger logger = LoggerFactory.getLogger(AudioCaptureService.class);

    private TargetDataLine line;
    private AudioFormat format;
    private boolean isRecording = false;
    private Thread recordingThread;
    private ByteArrayOutputStream audioData;

    // Default audio format: 16kHz, 16-bit, mono, signed, little-endian
    private static final float SAMPLE_RATE = 16000.0f;
    private static final int SAMPLE_SIZE_IN_BITS = 16;
    private static final int CHANNELS = 1;
    private static final boolean SIGNED = true;
    private static final boolean BIG_ENDIAN = false;

    // Buffer for reading from the capture line
    private byte[] buffer = new byte[4096];

    public AudioCaptureService() {
        initAudioFormat();
    }

    /**
     * Initialize the audio format to use for recording.
     */
    private void initAudioFormat() {
        format = new AudioFormat(
                SAMPLE_RATE,
                SAMPLE_SIZE_IN_BITS,
                CHANNELS,
                SIGNED,
                BIG_ENDIAN
        );
        logger.info("Initialized audio format: {}", format);
    }

    /**
     * Start audio capture from the default microphone.
     *
     * @return true if recording started successfully, false otherwise
     */
    public boolean startRecording() {
        logger.info("Starting audio recording");

        if (isRecording) {
            logger.warn("Recording is already in progress");
            return false;
        }

        try {
            // Get and open the target data line for capture
            DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);

            if (!AudioSystem.isLineSupported(info)) {
                logger.error("Line matching {} not supported", info);
                return false;
            }

            line = (TargetDataLine) AudioSystem.getLine(info);
            line.open(format);
            line.start();

            // Create a new audio data output stream
            audioData = new ByteArrayOutputStream();
            isRecording = true;

            // Start the recording thread
            recordingThread = new Thread(this::recordingLoop);
            recordingThread.start();

            logger.info("Recording started successfully");
            return true;

        } catch (LineUnavailableException e) {
            logger.error("Could not start recording", e);
            return false;
        }
    }

    /**
     * Stop the current recording.
     *
     * @return the recorded audio as a byte array
     */
    public byte[] stopRecording() {
        logger.info("Stopping audio recording");

        if (!isRecording) {
            logger.warn("No recording in progress");
            return new byte[0];
        }

        isRecording = false;

        try {
            // Wait for the recording thread to finish
            recordingThread.join(1000);
        } catch (InterruptedException e) {
            logger.warn("Interrupted while waiting for recording thread to finish", e);
            Thread.currentThread().interrupt();
        }

        line.stop();
        line.close();
        line = null;

        byte[] recordedData = audioData.toByteArray();
        logger.info("Recording stopped. Captured {} bytes", recordedData.length);

        return recordedData;
    }

    /**
     * The main recording loop that runs in a separate thread.
     */
    private void recordingLoop() {
        try {
            while (isRecording) {
                int bytesRead = line.read(buffer, 0, buffer.length);

                if (bytesRead > 0) {
                    audioData.write(buffer, 0, bytesRead);
                }
            }
        } catch (Exception e) {
            logger.error("Error in recording loop", e);
            isRecording = false;
        }
    }

    /**
     * Save the recorded audio to a WAV file.
     *
     * @param filePath the path to save the WAV file
     * @param audioBytes the audio data to save
     * @return true if saved successfully, false otherwise
     */
    public boolean saveToWavFile(String filePath, byte[] audioBytes) {
        File outputFile = new File(filePath);

        try (AudioInputStream ais = new AudioInputStream(
                new java.io.ByteArrayInputStream(audioBytes),
                format,
                audioBytes.length / format.getFrameSize())) {

            AudioSystem.write(ais, AudioFileFormat.Type.WAVE, outputFile);
            logger.info("Audio saved to: {}", outputFile.getAbsolutePath());
            return true;

        } catch (IOException e) {
            logger.error("Failed to save audio to WAV file", e);
            return false;
        }
    }

    /**
     * Get the current audio format used for recording.
     *
     * @return the audio format
     */
    public AudioFormat getAudioFormat() {
        return format;
    }

    /**
     * Check if recording is currently in progress.
     *
     * @return true if recording, false otherwise
     */
    public boolean isRecording() {
        return isRecording;
    }
}