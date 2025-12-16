package com.rednetty.willspeak.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rednetty.willspeak.model.TrainingPair;
import com.rednetty.willspeak.model.TrainingTemplate;
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
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Controller for training data collection functionality.
 */
public class TrainingDataController {
    private static final Logger logger = LoggerFactory.getLogger(TrainingDataController.class);

    private final AudioCaptureService audioCaptureService;
    private final SpeechProcessingService speechProcessingService;
    private final ProfileManager profileManager;
    private final ObjectMapper objectMapper;
    private final ExecutorService executorService;

    // Observable properties for UI binding
    private final ObservableList<UserProfile> profiles = FXCollections.observableArrayList();
    private final ObservableList<TrainingTemplate> templates = FXCollections.observableArrayList();
    private final ObservableList<JsonNode> trainingPairs = FXCollections.observableArrayList();
    private final ObjectProperty<UserProfile> selectedProfile = new SimpleObjectProperty<>();
    private final ObjectProperty<TrainingTemplate> selectedTemplate = new SimpleObjectProperty<>();
    private final StringProperty statusMessage = new SimpleStringProperty("Ready for training data collection");
    private final BooleanProperty recording = new SimpleBooleanProperty(false);
    private final BooleanProperty processing = new SimpleBooleanProperty(false);
    private final BooleanProperty trainingInProgress = new SimpleBooleanProperty(false);
    private final DoubleProperty trainingProgress = new SimpleDoubleProperty(0.0);

    private final HttpClient httpClient = HttpClient.newBuilder().build();

    // Paths for storing temporary recordings
    private File lastImpairedRecording;
    private File lastClearRecording;
    private String currentPrompt;
    private String recordingType; // "impaired" or "clear"
    private String lastRecordedFilePath;

    /**
     * Create a new TrainingDataController.
     *
     * @param serverUrl The URL of the WillSpeak server
     */
    public TrainingDataController(String serverUrl) {
        this.audioCaptureService = new AudioCaptureService();
        this.speechProcessingService = new SpeechProcessingService(serverUrl);
        this.profileManager = new ProfileManager();
        this.objectMapper = new ObjectMapper();
        this.executorService = Executors.newCachedThreadPool();

        // Initialize data
        loadProfiles();
        loadTemplates();
    }

    /**
     * Load user profiles from the profile manager.
     */
    public void loadProfiles() {
        profiles.clear();
        profiles.addAll(profileManager.getProfiles());
        logger.info("Loaded {} profiles", profiles.size());
    }

    /**
     * Load training templates from the server.
     */
    public void loadTemplates() {
        executorService.submit(() -> {
            try {
                CompletableFuture<JsonNode> future = speechProcessingService.getTemplates(null);
                JsonNode response = future.get();

                if (response.has("templates") && response.get("templates").isArray()) {
                    List<TrainingTemplate> loadedTemplates = new ArrayList<>();
                    for (JsonNode templateNode : response.get("templates")) {
                        String id = templateNode.get("id").asText();
                        String prompt = templateNode.get("prompt").asText();
                        String speakerName = templateNode.get("speaker_name").asText();
                        String category = templateNode.has("category") ? templateNode.get("category").asText() : "general";
                        String created = templateNode.get("created").asText();

                        TrainingTemplate template = new TrainingTemplate(id, prompt, speakerName, category, created);
                        loadedTemplates.add(template);
                    }

                    Platform.runLater(() -> {
                        templates.clear();
                        templates.addAll(loadedTemplates);
                        logger.info("Loaded {} templates", templates.size());
                    });
                }
            } catch (Exception e) {
                logger.error("Error loading templates", e);
                Platform.runLater(() -> setStatusMessage("Error loading templates: " + e.getMessage()));
            }
        });
    }

    /**
     * Load training pairs for the selected user.
     */
    public void loadUserTrainingPairs() {
        UserProfile profile = selectedProfile.get();
        if (profile == null) {
            trainingPairs.clear();
            return;
        }

        setStatusMessage("Loading training pairs...");
        executorService.submit(() -> {
            try {
                CompletableFuture<JsonNode> future = speechProcessingService.getUserTrainingPairs(profile.getId());
                JsonNode response = future.get();

                if (response.has("pairs") && response.get("pairs").isArray()) {
                    List<JsonNode> loadedPairs = new ArrayList<>();
                    for (JsonNode pairNode : response.get("pairs")) {
                        loadedPairs.add(pairNode);
                    }

                    Platform.runLater(() -> {
                        trainingPairs.clear();
                        trainingPairs.addAll(loadedPairs);
                        logger.info("Loaded {} training pairs for user {}", trainingPairs.size(), profile.getId());
                        setStatusMessage("Loaded " + trainingPairs.size() + " training pairs");
                    });
                }
            } catch (Exception e) {
                logger.error("Error loading training pairs", e);
                Platform.runLater(() -> setStatusMessage("Error loading training pairs: " + e.getMessage()));
            }
        });
    }

    /**
     * Upload a clear speech template to the server.
     *
     * @param prompt The text prompt that was spoken
     * @param speakerName The name of the speaker
     * @param category Optional category for organization
     * @param notes Optional notes about the speech pattern
     */
    public void uploadClearTemplate(String prompt, String speakerName, String category, String notes) {
        if (lastClearRecording == null || !lastClearRecording.exists()) {
            setStatusMessage("No clear speech recording available");
            return;
        }

        setProcessing(true);
        setStatusMessage("Uploading clear speech template...");

        executorService.submit(() -> {
            try {
                CompletableFuture<JsonNode> future = speechProcessingService.uploadClearTemplate(
                        lastClearRecording, prompt, speakerName, category);
                JsonNode response = future.get();

                // Add template to list
                String id = response.get("id").asText();
                String created = response.get("created").asText();
                TrainingTemplate template = new TrainingTemplate(id, prompt, speakerName, category, created);

                Platform.runLater(() -> {
                    templates.add(template);
                    setStatusMessage("Template uploaded successfully");
                    setProcessing(false);
                });
            } catch (Exception e) {
                logger.error("Error uploading template", e);
                Platform.runLater(() -> {
                    setStatusMessage("Error uploading template: " + e.getMessage());
                    setProcessing(false);
                });
            }
        });
    }

    /**
     * Create a training pair with both impaired and clear speech for the same prompt.
     *
     * @param prompt The text prompt that was spoken
     * @param notes Optional notes about the speech pattern
     */
    public void createTrainingPair(String prompt, String notes) {
        UserProfile profile = selectedProfile.get();
        if (profile == null) {
            setStatusMessage("No profile selected");
            return;
        }

        if (lastImpairedRecording == null || !lastImpairedRecording.exists()) {
            setStatusMessage("No impaired speech recording available");
            return;
        }

        if (lastClearRecording == null || !lastClearRecording.exists()) {
            setStatusMessage("No clear speech recording available");
            return;
        }

        setProcessing(true);
        setStatusMessage("Creating training pair...");

        executorService.submit(() -> {
            try {
                CompletableFuture<JsonNode> future = speechProcessingService.createTrainingPair(
                        lastImpairedRecording, lastClearRecording, prompt, profile.getId(), notes);
                JsonNode response = future.get();

                Platform.runLater(() -> {
                    trainingPairs.add(response);
                    setStatusMessage("Training pair created successfully");
                    setProcessing(false);
                });
            } catch (Exception e) {
                logger.error("Error creating training pair", e);
                Platform.runLater(() -> {
                    setStatusMessage("Error creating training pair: " + e.getMessage());
                    setProcessing(false);
                });
            }
        });
    }

    /**
     * Use a template for training with the user's impaired speech.
     *
     * @param notes Optional notes about the speech pattern
     */
    public void useTemplateForTraining(String notes) {
        UserProfile profile = selectedProfile.get();
        TrainingTemplate template = selectedTemplate.get();

        if (profile == null) {
            setStatusMessage("No profile selected");
            return;
        }

        if (template == null) {
            setStatusMessage("No template selected");
            return;
        }

        if (lastImpairedRecording == null || !lastImpairedRecording.exists()) {
            setStatusMessage("No impaired speech recording available");
            return;
        }

        setProcessing(true);
        setStatusMessage("Creating training pair from template...");

        executorService.submit(() -> {
            try {
                CompletableFuture<JsonNode> future = speechProcessingService.useTemplateForTraining(
                        lastImpairedRecording, template.getId(), profile.getId(), notes);
                JsonNode response = future.get();

                Platform.runLater(() -> {
                    trainingPairs.add(response);
                    setStatusMessage("Training pair created from template successfully");
                    setProcessing(false);
                });
            } catch (Exception e) {
                logger.error("Error creating training pair from template", e);
                Platform.runLater(() -> {
                    setStatusMessage("Error creating training pair: " + e.getMessage());
                    setProcessing(false);
                });
            }
        });
    }

    /**
     * Start recording audio for training pairs.
     *
     * @param type The type of recording ("impaired" or "clear")
     * @param prompt The text prompt to speak
     */
    public void startRecording(String type, String prompt) {
        if (audioCaptureService.isRecording()) {
            logger.warn("Already recording");
            return;
        }

        // Store the recording type for use when stopping
        this.recordingType = type;
        this.currentPrompt = prompt;

        if (audioCaptureService.startRecording()) {
            setRecording(true);  // Set the observable property to true
            setStatusMessage("Recording " + type + " speech. Say: \"" + prompt + "\"");
        } else {
            setStatusMessage("Failed to start recording");
        }
    }

    /**
     * Stop recording and save the audio.
     *
     * @return The path to the saved temporary file
     */
    public String stopRecording() {
        if (!audioCaptureService.isRecording()) {
            logger.warn("Not recording");
            return null;
        }

        byte[] audioData = audioCaptureService.stopRecording();
        setRecording(false);  // Set the observable property to false
        setStatusMessage("Recording stopped. Processing...");

        // Save audio to temporary file
        try {
            // Create temp file for audio
            Path tempFile = Files.createTempFile("training_" + recordingType + "_", ".wav");
            audioCaptureService.saveToWavFile(tempFile.toString(), audioData);

            // Store the file reference based on the type
            File audioFile = tempFile.toFile();
            if ("impaired".equals(recordingType)) {
                lastImpairedRecording = audioFile;
            } else if ("clear".equals(recordingType)) {
                lastClearRecording = audioFile;
            }

            // Store the file path
            this.lastRecordedFilePath = tempFile.toString();

            logger.info("Saved {} recording to: {}", recordingType, lastRecordedFilePath);
            return lastRecordedFilePath;
        } catch (Exception e) {
            logger.error("Error saving recording", e);
            setStatusMessage("Error saving recording: " + e.getMessage());
            return null;
        }
    }

    /**
     * Train model from collected training pairs.
     */
    public void trainFromPairs() {
        UserProfile profile = selectedProfile.get();
        if (profile == null) {
            setStatusMessage("No profile selected");
            return;
        }

        if (trainingPairs.isEmpty()) {
            setStatusMessage("No training pairs available");
            return;
        }

        setTrainingInProgress(true);
        setTrainingProgress(0.1); // Show some initial progress
        setStatusMessage("Starting model training...");

        executorService.submit(() -> {
            try {
                CompletableFuture<JsonNode> future = speechProcessingService.trainFromPairs(profile.getId());
                JsonNode response = future.get();

                String jobId = response.get("job_id").asText();
                logger.info("Started training job {} for user {}", jobId, profile.getId());

                // Update training status periodically
                monitorTrainingJob(jobId);

            } catch (Exception e) {
                logger.error("Error starting training", e);
                Platform.runLater(() -> {
                    setStatusMessage("Error starting training: " + e.getMessage());
                    setTrainingInProgress(false);
                    setTrainingProgress(0);
                });
            }
        });
    }

    /**
     * Monitor the progress of a training job.
     *
     * @param jobId The training job ID
     */
    private void monitorTrainingJob(String jobId) {
        executorService.submit(() -> {
            boolean completed = false;
            int attempts = 0;
            final int maxAttempts = 60; // 5 minutes with 5 second intervals

            while (!completed && attempts < maxAttempts) {
                try {
                    Thread.sleep(5000); // Check every 5 seconds

                    HttpRequest request = HttpRequest.newBuilder()
                            .uri(URI.create(speechProcessingService.getServerUrl() + "/training-data/training-jobs/" + jobId + "/status"))
                            .GET()
                            .build();

                    HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

                    if (response.statusCode() == 200) {
                        JsonNode jobInfo = objectMapper.readTree(response.body());
                        String status = jobInfo.get("status").asText();

                        double progress;
                        if ("running".equals(status)) {
                            progress = 0.3 + Math.min(0.6, attempts * 0.01); // Progress from 30% to 90% gradually
                        } else if ("completed".equals(status)) {
                            progress = 1.0;
                            completed = true;

                            Platform.runLater(() -> {
                                setStatusMessage("Training completed successfully!");
                                setTrainingProgress(progress);
                                setTrainingInProgress(false);

                                // Update user profile to reflect trained model
                                UserProfile profile = selectedProfile.get();
                                if (profile != null) {
                                    profile.setModelTrained(true);
                                    profileManager.saveProfile(profile);
                                }
                            });
                            return;
                        } else {
                            progress = 0.1;
                            if ("failed".equals(status)) {
                                String errorMessage = jobInfo.has("error") ? jobInfo.get("error").asText() : "Unknown error";
                                Platform.runLater(() -> {
                                    setStatusMessage("Training failed: " + errorMessage);
                                    setTrainingInProgress(false);
                                    setTrainingProgress(0);
                                });
                                return;
                            }
                        }

                        final double displayProgress = progress;
                        Platform.runLater(() -> {
                            setStatusMessage("Training in progress... " + Math.round(displayProgress * 100) + "%");
                            setTrainingProgress(displayProgress);
                        });
                    }

                    attempts++;
                } catch (Exception e) {
                    logger.error("Error monitoring training job", e);
                    attempts++;
                }
            }

            if (!completed) {
                Platform.runLater(() -> {
                    setStatusMessage("Training is taking longer than expected. Check back later.");
                    setTrainingInProgress(false);
                });
            }
        });
    }

    /**
     * Play a training pair audio file.
     *
     * @param pair The JSON data for the training pair to play
     * @param type The type of audio to play ("impaired" or "clear")
     */
    public void playTrainingPairAudio(JsonNode pair, String type) {
        setStatusMessage("Fetching audio file...");

        executorService.submit(() -> {
            try {
                String audioPath = "impaired".equals(type) ?
                        pair.path("impaired_path").asText() :
                        pair.path("clear_path").asText();

                // In a real implementation, you'd need to fetch the audio file from the server
                // For now, we'll just log the path
                logger.info("Would play audio from: {}", audioPath);

                Platform.runLater(() -> {
                    setStatusMessage("Audio playback not implemented in this version");
                });
            } catch (Exception e) {
                logger.error("Error playing audio", e);
                Platform.runLater(() -> {
                    setStatusMessage("Error playing audio: " + e.getMessage());
                });
            }
        });
    }

    /**
     * Clean up resources.
     */
    public void shutdown() {
        if (audioCaptureService.isRecording()) {
            audioCaptureService.stopRecording();
        }

        executorService.shutdown();

        // Clean up temp files
        if (lastImpairedRecording != null && lastImpairedRecording.exists()) {
            lastImpairedRecording.delete();
        }
        if (lastClearRecording != null && lastClearRecording.exists()) {
            lastClearRecording.delete();
        }
    }

    // Observable property getters and setters

    public ObservableList<UserProfile> getProfiles() {
        return profiles;
    }

    public ObservableList<TrainingTemplate> getTemplates() {
        return templates;
    }

    public ObservableList<JsonNode> getTrainingPairs() {
        return trainingPairs;
    }

    public UserProfile getSelectedProfile() {
        return selectedProfile.get();
    }

    public ObjectProperty<UserProfile> selectedProfileProperty() {
        return selectedProfile;
    }

    public void setSelectedProfile(UserProfile profile) {
        this.selectedProfile.set(profile);
        if (profile != null) {
            loadUserTrainingPairs();
        }
    }

    public TrainingTemplate getSelectedTemplate() {
        return selectedTemplate.get();
    }

    public ObjectProperty<TrainingTemplate> selectedTemplateProperty() {
        return selectedTemplate;
    }

    public void setSelectedTemplate(TrainingTemplate template) {
        this.selectedTemplate.set(template);
    }

    public String getStatusMessage() {
        return statusMessage.get();
    }

    public StringProperty statusMessageProperty() {
        return statusMessage;
    }

    public void setStatusMessage(String message) {
        this.statusMessage.set(message);
    }

    public boolean isRecording() {
        return recording.get();
    }

    public BooleanProperty recordingProperty() {
        return recording;
    }

    public void setRecording(boolean recording) {
        this.recording.set(recording);
    }

    public boolean isProcessing() {
        return processing.get();
    }

    public BooleanProperty processingProperty() {
        return processing;
    }

    public void setProcessing(boolean processing) {
        this.processing.set(processing);
    }

    public boolean isTrainingInProgress() {
        return trainingInProgress.get();
    }

    public BooleanProperty trainingInProgressProperty() {
        return trainingInProgress;
    }

    public void setTrainingInProgress(boolean trainingInProgress) {
        this.trainingInProgress.set(trainingInProgress);
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

    /**
     * Get the path to the last recorded file.
     *
     * @return The file path as a string
     */
    public String getLastRecordedFilePath() {
        return this.lastRecordedFilePath;
    }

    /**
     * Set the impaired recording file.
     *
     * @param file The file to set as impaired recording
     */
    public void setImpairedRecording(File file) {
        this.lastImpairedRecording = file;
    }

    /**
     * Set the clear recording file.
     *
     * @param file The file to set as clear recording
     */
    public void setClearRecording(File file) {
        this.lastClearRecording = file;
    }

    /**
     * Get the last impaired recording file.
     *
     * @return The impaired recording file
     */
    public File getImpairedRecording() {
        return this.lastImpairedRecording;
    }

    /**
     * Get the last clear recording file.
     *
     * @return The clear recording file
     */
    public File getClearRecording() {
        return this.lastClearRecording;
    }
}