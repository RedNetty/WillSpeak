package com.rednetty.willspeak.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rednetty.willspeak.model.TrainingSession;
import com.rednetty.willspeak.model.UserProfile;
import com.rednetty.willspeak.service.AudioCaptureService;
import com.rednetty.willspeak.service.ProfileManager;
import com.rednetty.willspeak.service.SpeechProcessingService;
import javafx.application.Platform;
import javafx.beans.property.*;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Controller for training functionality.
 */
public class TrainingController {
    private static final Logger logger = LoggerFactory.getLogger(TrainingController.class);

    private final AudioCaptureService audioCaptureService;
    private final SpeechProcessingService speechProcessingService;
    private final ProfileManager profileManager;
    private final ObjectMapper objectMapper;
    private final ExecutorService executorService;

    // Observable properties for UI binding
    private final ObservableList<UserProfile> profiles = FXCollections.observableArrayList();
    private final ObjectProperty<UserProfile> selectedProfile = new SimpleObjectProperty<>();
    private final StringProperty statusMessage = new SimpleStringProperty("Ready for training");
    private final BooleanProperty training = new SimpleBooleanProperty(false);
    private final DoubleProperty trainingProgress = new SimpleDoubleProperty(0.0);

    // Training session state
    private String currentSessionId;
    private List<String> trainingPrompts;
    private int currentPromptIndex;
    private byte[] lastRecordedAudio;

    public TrainingController(String serverUrl) {
        this.audioCaptureService = new AudioCaptureService();
        this.speechProcessingService = new SpeechProcessingService(serverUrl);
        this.profileManager = new ProfileManager();
        this.objectMapper = new ObjectMapper();
        this.executorService = Executors.newCachedThreadPool();

        // Initialize training prompts
        initTrainingPrompts();

        // Load profiles
        loadProfiles();
    }

    /**
     * Initialize standard training prompts.
     */
    private void initTrainingPrompts() {
        trainingPrompts = new ArrayList<>();
        trainingPrompts.add("My name is John Smith");
        trainingPrompts.add("Today is a beautiful day");
        trainingPrompts.add("Please turn on the lights");
        trainingPrompts.add("I would like a glass of water");
        trainingPrompts.add("Could you help me with this");
        trainingPrompts.add("Thank you very much for your help");
        trainingPrompts.add("What time is the meeting today");
        trainingPrompts.add("I'm feeling great this morning");
        trainingPrompts.add("The quick brown fox jumps over the lazy dog");
        trainingPrompts.add("She sells seashells by the seashore");
    }

    /**
     * Load user profiles.
     */
    public void loadProfiles() {
        profiles.clear();
        profiles.addAll(profileManager.getProfiles());
        logger.info("Loaded {} profiles", profiles.size());
    }

    /**
     * Create a new user profile.
     *
     * @param name The name for the new profile
     * @return The created profile
     */
    public UserProfile createProfile(String name) {
        UserProfile profile = profileManager.createProfile(name);
        profiles.add(profile);
        return profile;
    }

    /**
     * Create a new user profile on the server.
     *
     * @param name The name for the new profile
     * @param description Optional description
     */
    public void createServerProfile(String name, String description) {
        executorService.submit(() -> {
            try {
                // Build request
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(speechProcessingService.getServerUrl() + "/user/create"))
                        .header("Content-Type", "application/x-www-form-urlencoded")
                        .POST(HttpRequest.BodyPublishers.ofString(
                                "name=" + name + "&description=" + (description != null ? description : "")))
                        .build();

                // Send request
                HttpClient client = HttpClient.newHttpClient();
                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    JsonNode json = objectMapper.readTree(response.body());
                    String userId = json.get("id").asText();

                    // Create local profile
                    UserProfile profile = createProfile(name);
                    profile.setDescription(description != null ? description : "");
                    profileManager.saveProfile(profile);

                    Platform.runLater(() -> {
                        setStatusMessage("Profile created: " + name);
                        loadProfiles();
                    });
                } else {
                    logger.error("Error creating server profile: {}", response.body());
                    Platform.runLater(() ->
                            setStatusMessage("Error creating profile: " + response.statusCode())
                    );
                }

            } catch (Exception e) {
                logger.error("Error creating server profile", e);
                Platform.runLater(() ->
                        setStatusMessage("Error creating profile: " + e.getMessage())
                );
            }
        });
    }

    /**
     * Start a new training session.
     *
     * @return true if started successfully, false otherwise
     */
    public boolean startTrainingSession() {
        if (training.get()) {
            logger.warn("Training session already in progress");
            return false;
        }

        UserProfile profile = selectedProfile.get();
        if (profile == null) {
            setStatusMessage("No profile selected");
            return false;
        }

        // Reset training state
        currentPromptIndex = 0;
        trainingProgress.set(0.0);

        // Start training session on server
        executorService.submit(() -> {
            try {
                // Build request
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(speechProcessingService.getServerUrl() + "/training/start"))
                        .header("Content-Type", "application/x-www-form-urlencoded")
                        .POST(HttpRequest.BodyPublishers.ofString("user_id=" + profile.getId()))
                        .build();

                // Send request
                HttpClient client = HttpClient.newHttpClient();
                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    JsonNode json = objectMapper.readTree(response.body());
                    currentSessionId = json.get("id").asText();

                    Platform.runLater(() -> {
                        setTraining(true);
                        setStatusMessage("Training session started. Say: \"" + getCurrentPrompt() + "\"");
                    });
                } else {
                    logger.error("Error starting training session: {}", response.body());
                    Platform.runLater(() ->
                            setStatusMessage("Error starting training: " + response.statusCode())
                    );
                }

            } catch (Exception e) {
                logger.error("Error starting training session", e);
                Platform.runLater(() ->
                        setStatusMessage("Error starting training: " + e.getMessage())
                );
            }
        });

        return true;
    }

    /**
     * Record audio for the current training prompt.
     */
    public void recordTrainingAudio() {
        if (!training.get()) {
            logger.warn("No training session in progress");
            return;
        }

        if (audioCaptureService.isRecording()) {
            logger.warn("Already recording");
            return;
        }

        if (audioCaptureService.startRecording()) {
            setStatusMessage("Recording... Say: \"" + getCurrentPrompt() + "\"");
        } else {
            setStatusMessage("Failed to start recording");
        }
    }

    /**
     * Stop recording and save the training sample.
     */
    public void stopRecordingAndSave() {
        if (!audioCaptureService.isRecording()) {
            logger.warn("Not recording");
            return;
        }

        lastRecordedAudio = audioCaptureService.stopRecording();
        setStatusMessage("Recording stopped. Processing...");

        // Save training sample
        executorService.submit(() -> {
            try {
                // Create temp file for audio
                Path tempFile = Files.createTempFile("training_", ".wav");
                audioCaptureService.saveToWavFile(tempFile.toString(), lastRecordedAudio);

                // Prepare multipart request
                String boundary = "WillSpeakBoundary" + System.currentTimeMillis();

                // Build request body
                String prompt = getCurrentPrompt();
                File audioFile = tempFile.toFile();

                // Upload to server
                CompletableFuture<JsonNode> future = speechProcessingService.uploadTrainingAudio(
                        currentSessionId, prompt, audioFile);

                future.thenAccept(json -> {
                    // Update progress
                    currentPromptIndex++;
                    double progress = (double) currentPromptIndex / trainingPrompts.size();

                    Platform.runLater(() -> {
                        trainingProgress.set(progress);

                        if (currentPromptIndex < trainingPrompts.size()) {
                            setStatusMessage("Sample recorded. Next prompt: \"" + getCurrentPrompt() + "\"");
                        } else {
                            completeTrainingSession();
                        }
                    });

                    // Clean up temp file
                    try {
                        Files.deleteIfExists(tempFile);
                    } catch (IOException e) {
                        logger.warn("Failed to delete temp file", e);
                    }
                }).exceptionally(e -> {
                    logger.error("Error saving training sample", e);
                    Platform.runLater(() ->
                            setStatusMessage("Error saving sample: " + e.getMessage())
                    );
                    return null;
                });

            } catch (Exception e) {
                logger.error("Error processing training audio", e);
                Platform.runLater(() ->
                        setStatusMessage("Error processing audio: " + e.getMessage())
                );
            }
        });
    }

    /**
     * Complete the training session.
     */
    private void completeTrainingSession() {
        if (currentSessionId == null) {
            logger.warn("No active training session");
            return;
        }

        setStatusMessage("Training session complete. Processing data...");

        executorService.submit(() -> {
            try {
                // Build request
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(speechProcessingService.getServerUrl() +
                                "/training/" + currentSessionId + "/complete"))
                        .POST(HttpRequest.BodyPublishers.noBody())
                        .build();

                // Send request
                HttpClient client = HttpClient.newHttpClient();
                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    // Update local profile
                    UserProfile profile = selectedProfile.get();

                    if (profile != null) {
                        // Create training session record
                        TrainingSession session = new TrainingSession();
                        profile.addTrainingSession(session);
                        profileManager.saveProfile(profile);
                    }

                    Platform.runLater(() -> {
                        setTraining(false);
                        setStatusMessage("Training complete! Model is being trained in the background.");
                    });
                } else {
                    logger.error("Error completing training session: {}", response.body());
                    Platform.runLater(() -> {
                        setTraining(false);
                        setStatusMessage("Error completing training: " + response.statusCode());
                    });
                }

            } catch (Exception e) {
                logger.error("Error completing training session", e);
                Platform.runLater(() -> {
                    setTraining(false);
                    setStatusMessage("Error completing training: " + e.getMessage());
                });
            }
        });
    }

    /**
     * Get the current training prompt.
     *
     * @return The current prompt
     */
    public String getCurrentPrompt() {
        if (currentPromptIndex < 0 || currentPromptIndex >= trainingPrompts.size()) {
            return "";
        }
        return trainingPrompts.get(currentPromptIndex);
    }

    /**
     * Cancel the current training session.
     */
    public void cancelTrainingSession() {
        if (!training.get()) {
            return;
        }

        if (audioCaptureService.isRecording()) {
            audioCaptureService.stopRecording();
        }

        setTraining(false);
        setStatusMessage("Training canceled");
        currentSessionId = null;
    }

    /**
     * Clean up resources.
     */
    public void shutdown() {
        if (audioCaptureService.isRecording()) {
            audioCaptureService.stopRecording();
        }

        executorService.shutdown();
    }

    // Observable property getters and setters

    public ObservableList<UserProfile> getProfiles() {
        return profiles;
    }

    public UserProfile getSelectedProfile() {
        return selectedProfile.get();
    }

    public ObjectProperty<UserProfile> selectedProfileProperty() {
        return selectedProfile;
    }

    public void setSelectedProfile(UserProfile selectedProfile) {
        this.selectedProfile.set(selectedProfile);
    }

    public String getStatusMessage() {
        return statusMessage.get();
    }

    public StringProperty statusMessageProperty() {
        return statusMessage;
    }

    public void setStatusMessage(String statusMessage) {
        this.statusMessage.set(statusMessage);
    }

    public boolean isTraining() {
        return training.get();
    }

    public BooleanProperty trainingProperty() {
        return training;
    }

    public void setTraining(boolean training) {
        this.training.set(training);
    }

    public double getTrainingProgress() {
        return trainingProgress.get();
    }

    public DoubleProperty trainingProgressProperty() {
        return trainingProgress;
    }

    public void setTrainingProgress(double trainingProgress) {
        this.trainingProgress.set(trainingProgress);
    }
}