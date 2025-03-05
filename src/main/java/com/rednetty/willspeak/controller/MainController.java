package com.rednetty.willspeak.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.rednetty.willspeak.service.AudioCaptureService;
import com.rednetty.willspeak.service.SpeechProcessingService;
import javafx.application.Platform;
import javafx.beans.property.*;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Controller for the main application functions.
 */
public class MainController {
    private static final Logger logger = LoggerFactory.getLogger(MainController.class);

    private final AudioCaptureService audioCaptureService;
    private final SpeechProcessingService speechProcessingService;
    private final ExecutorService executorService;

    // Observable properties for UI binding
    private final BooleanProperty recording = new SimpleBooleanProperty(false);
    private final BooleanProperty processing = new SimpleBooleanProperty(false);
    private final BooleanProperty serverConnected = new SimpleBooleanProperty(false);
    private final BooleanProperty realTimeMode = new SimpleBooleanProperty(false);
    private final StringProperty transcriptionText = new SimpleStringProperty("");
    private final StringProperty statusMessage = new SimpleStringProperty("Ready");

    // Store recorded audio for processing
    private byte[] lastRecordedAudio;
    private byte[] lastProcessedAudio;

    private MediaPlayer mediaPlayer;

    public MainController(String serverUrl) {
        this.audioCaptureService = new AudioCaptureService();
        this.speechProcessingService = new SpeechProcessingService(serverUrl);
        this.executorService = Executors.newCachedThreadPool();

        // Check server connection on startup
        checkServerConnection();
    }

    /**
     * Start recording audio.
     */
    public void startRecording() {
        if (audioCaptureService.startRecording()) {
            setRecording(true);
            setStatusMessage("Recording...");
        } else {
            setStatusMessage("Failed to start recording");
        }
    }

    /**
     * Stop recording and get the recorded audio.
     */
    public void stopRecording() {
        if (audioCaptureService.isRecording()) {
            lastRecordedAudio = audioCaptureService.stopRecording();
            setRecording(false);
            setStatusMessage("Recording stopped. " + lastRecordedAudio.length + " bytes captured.");
        }
    }

    /**
     * Process the last recorded audio through the server.
     *
     * @param userId The user ID to process with (may be null)
     */
    public void processRecordedAudio(String userId) {
        if (lastRecordedAudio == null || lastRecordedAudio.length == 0) {
            setStatusMessage("No recording available to process");
            return;
        }

        setProcessing(true);
        setStatusMessage("Processing audio...");

        executorService.submit(() -> {
            try {
                JsonNode response = speechProcessingService.processAudio(lastRecordedAudio, userId);

                // Update UI with results
                Platform.runLater(() -> {
                    String transcription = response.path("transcription").asText("");
                    setTranscriptionText(transcription);
                    setStatusMessage("Processing complete");

                    // Save processed audio if needed
                    if (response.has("output_file")) {
                        // In a real implementation, we'd download and store the processed file
                        // For now, we'll just use the original audio for playback
                        lastProcessedAudio = lastRecordedAudio;
                    }
                    setProcessing(false);
                });
            } catch (Exception e) {
                logger.error("Error processing audio", e);
                Platform.runLater(() -> {
                    setStatusMessage("Error processing audio: " + e.getMessage());
                    setProcessing(false);
                });
            }
        });
    }

    /**
     * Play the last processed audio.
     */
    public void playProcessedAudio() {
        byte[] audioToPlay = lastProcessedAudio != null ? lastProcessedAudio : lastRecordedAudio;

        if (audioToPlay == null || audioToPlay.length == 0) {
            setStatusMessage("No audio available to play");
            return;
        }

        try {
            // Create a temporary WAV file for playback
            Path tempFile = Files.createTempFile("willspeak_playback_", ".wav");

            // Save the audio to the temp file
            AudioFormat format = audioCaptureService.getAudioFormat();
            AudioInputStream ais = new AudioInputStream(
                    new ByteArrayInputStream(audioToPlay),
                    format,
                    audioToPlay.length / format.getFrameSize());

            AudioSystem.write(ais, javax.sound.sampled.AudioFileFormat.Type.WAVE, tempFile.toFile());

            // Play using JavaFX MediaPlayer
            Media media = new Media(tempFile.toUri().toString());

            if (mediaPlayer != null) {
                mediaPlayer.stop();
                mediaPlayer.dispose();
            }

            mediaPlayer = new MediaPlayer(media);
            mediaPlayer.setOnEndOfMedia(() -> {
                try {
                    Files.deleteIfExists(tempFile);
                } catch (IOException e) {
                    logger.warn("Failed to delete temp file", e);
                }
            });

            mediaPlayer.play();
            setStatusMessage("Playing audio...");

        } catch (IOException e) {
            logger.error("Error playing audio", e);
            setStatusMessage("Error playing audio: " + e.getMessage());
        }
    }

    /**
     * Start real-time audio processing mode.
     *
     * @param userId The user ID to process with (may be null)
     */
    public void startRealTimeProcessing(String userId) {
        // First ensure we're not recording
        if (audioCaptureService.isRecording()) {
            audioCaptureService.stopRecording();
        }

        // Set up WebSocket for real-time processing
        boolean connected = speechProcessingService.startRealTimeProcessing(processedBytes -> {
            // This consumer receives audio data from the server in real-time
            // TODO: Route the processed audio to the audio output device
            logger.debug("Received {} bytes of processed audio", processedBytes.length);
        }, userId);

        if (connected) {
            // Start capturing audio again
            if (audioCaptureService.startRecording()) {
                setRecording(true);
                setRealTimeMode(true);
                setStatusMessage("Real-time processing active");

                // Start a thread to send audio chunks to the server
                executorService.submit(this::realTimeProcessingLoop);
            } else {
                speechProcessingService.stopRealTimeProcessing();
                setStatusMessage("Failed to start audio capture for real-time processing");
            }
        } else {
            setStatusMessage("Failed to establish real-time connection to server");
        }
    }

    /**
     * Stop real-time audio processing.
     */
    public void stopRealTimeProcessing() {
        if (audioCaptureService.isRecording()) {
            audioCaptureService.stopRecording();
            setRecording(false);
        }

        speechProcessingService.stopRealTimeProcessing();
        setRealTimeMode(false);
        setStatusMessage("Real-time processing stopped");
    }

    /**
     * Check if the server is available.
     */
    public void checkServerConnection() {
        executorService.submit(() -> {
            boolean connected = speechProcessingService.checkServerConnection();
            Platform.runLater(() -> {
                setServerConnected(connected);
                setStatusMessage(connected ? "Connected to server" : "Server not available");
            });
        });
    }

    /**
     * Background thread for real-time processing.
     */
    private void realTimeProcessingLoop() {
        byte[] buffer = new byte[4096]; // Same size as in AudioCaptureService

        try {
            while (audioCaptureService.isRecording() && realTimeMode.get()) {
                // In a real implementation, we would get chunks directly from the AudioCaptureService
                // For now, this is just a placeholder
                Thread.sleep(100); // Sleep to avoid busy waiting

                // Process audio chunks
                speechProcessingService.processAudioChunk(buffer);
            }
        } catch (InterruptedException e) {
            logger.warn("Real-time processing interrupted", e);
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            logger.error("Error in real-time processing loop", e);
            Platform.runLater(() -> setStatusMessage("Error in real-time processing: " + e.getMessage()));
        } finally {
            // Make sure we stop everything properly
            Platform.runLater(this::stopRealTimeProcessing);
        }
    }

    /**
     * Get the last recorded audio data.
     *
     * @return The last recorded audio as byte array
     */
    public byte[] getLastRecordedAudio() {
        return lastRecordedAudio;
    }

    /**
     * Save the last recorded audio to a file.
     *
     * @param filePath The path to save the audio file
     * @return True if saved successfully, false otherwise
     */
    public boolean saveLastRecordedAudio(String filePath) {
        if (lastRecordedAudio == null || lastRecordedAudio.length == 0) {
            return false;
        }
        return audioCaptureService.saveToWavFile(filePath, lastRecordedAudio);
    }

    /**
     * Clean up resources on application exit.
     */
    public void shutdown() {
        if (audioCaptureService.isRecording()) {
            audioCaptureService.stopRecording();
        }

        speechProcessingService.stopRealTimeProcessing();
        executorService.shutdown();

        if (mediaPlayer != null) {
            mediaPlayer.stop();
            mediaPlayer.dispose();
        }
    }

    // Getters and setters for observable properties

    public boolean isRecording() {
        return recording.get();
    }

    public BooleanProperty recordingProperty() {
        return recording;
    }

    private void setRecording(boolean recording) {
        this.recording.set(recording);
    }

    public boolean isProcessing() {
        return processing.get();
    }

    public BooleanProperty processingProperty() {
        return processing;
    }

    private void setProcessing(boolean processing) {
        this.processing.set(processing);
    }

    public boolean isServerConnected() {
        return serverConnected.get();
    }

    public BooleanProperty serverConnectedProperty() {
        return serverConnected;
    }

    private void setServerConnected(boolean serverConnected) {
        this.serverConnected.set(serverConnected);
    }

    public boolean isRealTimeMode() {
        return realTimeMode.get();
    }

    public BooleanProperty realTimeModeProperty() {
        return realTimeMode;
    }

    private void setRealTimeMode(boolean realTimeMode) {
        this.realTimeMode.set(realTimeMode);
    }

    public String getTranscriptionText() {
        return transcriptionText.get();
    }

    public StringProperty transcriptionTextProperty() {
        return transcriptionText;
    }

    private void setTranscriptionText(String transcriptionText) {
        this.transcriptionText.set(transcriptionText);
    }

    public String getStatusMessage() {
        return statusMessage.get();
    }

    public StringProperty statusMessageProperty() {
        return statusMessage;
    }

    private void setStatusMessage(String statusMessage) {
        this.statusMessage.set(statusMessage);
    }
}