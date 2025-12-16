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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

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
    private final HttpClient httpClient = HttpClient.newBuilder().build();
    // Observable properties for UI binding
    private final ObservableList<UserProfile> profiles = FXCollections.observableArrayList();
    private final ObjectProperty<UserProfile> selectedProfile = new SimpleObjectProperty<>();
    private final StringProperty statusMessage = new SimpleStringProperty("Ready for training");
    private final BooleanProperty training = new SimpleBooleanProperty(false);
    private final DoubleProperty trainingProgress = new SimpleDoubleProperty(0.0);

    // Training pair related properties
    private final ObservableList<JsonNode> trainingPairs = FXCollections.observableArrayList();
    private final BooleanProperty pairsTraining = new SimpleBooleanProperty(false);
    private ScheduledExecutorService statusPollingScheduler;

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
        setStatusMessage("Creating profile: " + name + "...");

        executorService.submit(() -> {
            try {
                logger.info("Creating server profile for: {}", name);

                // Create the profile on the server and wait for the response
                CompletableFuture<JsonNode> future = speechProcessingService.createUserProfile(name, description);
                JsonNode json = future.get();

                // Extract the server-generated ID
                String userId = json.get("id").asText();
                logger.info("Server created profile with ID: {}", userId);

                // Create local profile with the server ID
                UserProfile profile = new UserProfile(userId, name);
                profile.setDescription(description != null ? description : "");
                profileManager.saveProfile(profile);
                logger.info("Local profile saved with server ID: {}", userId);

                Platform.runLater(() -> {
                    // First refresh our profiles list
                    loadProfiles();
                    setStatusMessage("Profile created: " + name);

                    // Note: Don't try to directly set the selectedProfile as it might be bound to UI controls
                });
            } catch (InterruptedException e) {
                logger.error("Server profile creation interrupted", e);
                Thread.currentThread().interrupt();
                Platform.runLater(() ->
                        setStatusMessage("Error creating profile: Operation interrupted")
                );
            } catch (ExecutionException e) {
                logger.error("Error creating server profile", e.getCause());
                Platform.runLater(() ->
                        setStatusMessage("Error creating profile: " + e.getCause().getMessage())
                );
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
        setStatusMessage("Starting training session for " + profile.getName() + "...");

        // Log the profile information for debugging
        logger.info("Selected profile for training: {} (ID: {})", profile.getName(), profile.getId());

        // Start training session on server
        executorService.submit(() -> {
            try {
                // Get the server-generated user ID from our local profile
                String userId = profile.getId();
                logger.info("Starting training session for user ID: {}", userId);

                // First verify user exists on server
                CompletableFuture<Boolean> checkFuture = verifyUserExists(userId);
                Boolean userExists = checkFuture.get();

                if (!userExists) {
                    throw new RuntimeException("Server cannot find user with ID: " + userId);
                }

                // Use the SpeechProcessingService to start a training session with verified user
                CompletableFuture<JsonNode> future = speechProcessingService.startTrainingSession(userId);
                JsonNode response = future.get();

                // Extract session ID from response
                currentSessionId = response.get("id").asText();
                logger.info("Training session started with ID: {}", currentSessionId);

                Platform.runLater(() -> {
                    setTraining(true);
                    setStatusMessage("Training session started. Say: \"" + getCurrentPrompt() + "\"");
                });
            } catch (InterruptedException e) {
                logger.error("Training session start interrupted", e);
                Thread.currentThread().interrupt();
                Platform.runLater(() ->
                        setStatusMessage("Error starting training: Operation interrupted")
                );
            } catch (ExecutionException e) {
                logger.error("Error starting training session: {}", e.getCause() != null ? e.getCause().getMessage() : e.getMessage(), e);

                String errorMessage = e.getCause() != null ? e.getCause().getMessage() : e.getMessage();
                if (errorMessage.contains("User not found")) {
                    errorMessage = "Server cannot find the user profile. Try creating a new profile.";
                }

                final String finalErrorMessage = errorMessage;
                Platform.runLater(() ->
                        setStatusMessage("Error starting training: " + finalErrorMessage)
                );
            } catch (Exception e) {
                logger.error("Error starting training session: {}", e.getMessage(), e);
                Platform.runLater(() ->
                        setStatusMessage("Error starting training: " + e.getMessage())
                );
            }
        });

        return true;
    }

    /**
     * Verify if a user exists on the server.
     *
     * @param userId The user ID to check
     * @return CompletableFuture<Boolean> that resolves to true if the user exists
     */
    private CompletableFuture<Boolean> verifyUserExists(String userId) {
        CompletableFuture<Boolean> future = new CompletableFuture<>();

        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(speechProcessingService.getServerUrl() + "/user/" + userId))
                    .GET()
                    .build();

            httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                    .thenApply(response -> {
                        logger.info("User verification response for ID {}: {} - {}",
                                userId, response.statusCode(),
                                response.statusCode() == 200 ? "User exists" : response.body());
                        return response.statusCode() == 200;
                    })
                    .whenComplete((result, error) -> {
                        if (error != null) {
                            logger.error("Error verifying user", error);
                            future.complete(false);
                        } else {
                            future.complete(result);
                        }
                    });
        } catch (Exception e) {
            logger.error("Error preparing user verification request", e);
            future.complete(false);
        }

        return future;
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

                // Get current prompt
                String prompt = getCurrentPrompt();
                File audioFile = tempFile.toFile();

                logger.info("Uploading audio sample for prompt: \"{}\"", prompt);

                // Upload to server
                CompletableFuture<JsonNode> future = speechProcessingService.uploadTrainingAudio(
                        currentSessionId, prompt, audioFile);

                future.thenAccept(json -> {
                    logger.info("Successfully uploaded training sample: {}", json);

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
                logger.info("Completing training session: {}", currentSessionId);

                // Complete the training session on the server
                CompletableFuture<JsonNode> future = speechProcessingService.completeTrainingSession(currentSessionId);
                JsonNode response = future.get();
                logger.info("Training session completed successfully: {}", response);

                // Update local profile
                UserProfile profile = selectedProfile.get();

                if (profile != null) {
                    // Create training session record
                    TrainingSession session = new TrainingSession();
                    profile.addTrainingSession(session);
                    profileManager.saveProfile(profile);
                    logger.info("Updated local profile with training session");
                }

                Platform.runLater(() -> {
                    setTraining(false);
                    setStatusMessage("Training complete! Model is being trained in the background.");
                });
            } catch (InterruptedException e) {
                logger.error("Training session completion interrupted", e);
                Thread.currentThread().interrupt();
                Platform.runLater(() -> {
                    setTraining(false);
                    setStatusMessage("Error completing training: Operation interrupted");
                });
            } catch (ExecutionException e) {
                logger.error("Error completing training session", e.getCause());
                Platform.runLater(() -> {
                    setTraining(false);
                    setStatusMessage("Error completing training: " + e.getCause().getMessage());
                });
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
     * Fetch training pairs for the currently selected user.
     */
    public void loadTrainingPairs() {
        UserProfile profile = selectedProfile.get();
        if (profile == null) {
            setStatusMessage("No profile selected");
            return;
        }

        String userId = profile.getId();
        setStatusMessage("Loading training pairs...");

        executorService.submit(() -> {
            try {
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(speechProcessingService.getServerUrl() + "/training-data/user-pairs/" + userId))
                        .GET()
                        .build();

                HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    JsonNode responseJson = objectMapper.readTree(response.body());
                    JsonNode pairs = responseJson.get("pairs");

                    Platform.runLater(() -> {
                        trainingPairs.clear();
                        if (pairs != null && pairs.isArray()) {
                            for (JsonNode pair : pairs) {
                                trainingPairs.add(pair);
                            }
                        }
                        setStatusMessage("Loaded " + trainingPairs.size() + " training pairs");
                    });
                } else {
                    Platform.runLater(() -> {
                        setStatusMessage("Error loading training pairs: " + response.body());
                    });
                }
            } catch (Exception e) {
                logger.error("Error loading training pairs", e);
                Platform.runLater(() -> {
                    setStatusMessage("Error loading training pairs: " + e.getMessage());
                });
            }
        });
    }

    /**
     * Start training from collected pairs.
     */
    public void startTrainingFromPairs() {
        UserProfile profile = selectedProfile.get();
        if (profile == null) {
            setStatusMessage("No profile selected");
            return;
        }

        if (trainingPairs.isEmpty()) {
            setStatusMessage("No training pairs available. Create some pairs first.");
            return;
        }

        if (pairsTraining.get()) {
            setStatusMessage("Training already in progress");
            return;
        }

        String userId = profile.getId();
        setStatusMessage("Starting model training from " + trainingPairs.size() + " collected pairs...");
        setPairsTraining(true);

        executorService.submit(() -> {
            try {
                // Create form data
                String boundary = "WillSpeakBoundary" + System.currentTimeMillis();
                ByteArrayOutputStream requestBody = new ByteArrayOutputStream();

                // Add user_id part
                String userIdPartHeader =
                        "--" + boundary + "\r\n" +
                                "Content-Disposition: form-data; name=\"user_id\"\r\n\r\n";
                requestBody.write(userIdPartHeader.getBytes());
                requestBody.write(userId.getBytes());
                requestBody.write(("\r\n--" + boundary + "--\r\n").getBytes());

                // Create HTTP request
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(speechProcessingService.getServerUrl() + "/training-data/train-from-pairs"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(requestBody.toByteArray()))
                        .build();

                // Send request
                HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    JsonNode responseJson = objectMapper.readTree(response.body());
                    String jobId = responseJson.get("job_id").asText();

                    Platform.runLater(() -> {
                        setStatusMessage("Training started successfully. Job ID: " + jobId);
                    });

                    // Start polling for status
                    pollTrainingStatus(jobId);
                } else {
                    Platform.runLater(() -> {
                        setPairsTraining(false);
                        setStatusMessage("Failed to start training: " + response.body());
                    });
                }
            } catch (Exception e) {
                logger.error("Error starting training from pairs", e);
                Platform.runLater(() -> {
                    setPairsTraining(false);
                    setStatusMessage("Error starting training: " + e.getMessage());
                });
            }
        });
    }

    /**
     * Poll the training job status.
     *
     * @param jobId The job ID to poll
     */
    private void pollTrainingStatus(String jobId) {
        // Cancel any existing polling
        if (statusPollingScheduler != null && !statusPollingScheduler.isShutdown()) {
            statusPollingScheduler.shutdownNow();
        }

        // Create new polling scheduler
        statusPollingScheduler = Executors.newScheduledThreadPool(1);
        statusPollingScheduler.scheduleAtFixedRate(() -> {
            try {
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(speechProcessingService.getServerUrl() + "/training-data/training-jobs/" + jobId + "/status"))
                        .GET()
                        .build();

                HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    JsonNode job = objectMapper.readTree(response.body());
                    String status = job.get("status").asText();

                    Platform.runLater(() -> {
                        setStatusMessage("Training status: " + status);
                    });

                    // Stop polling when complete or failed
                    if (status.equals("completed") || status.equals("failed")) {
                        if (status.equals("completed")) {
                            Platform.runLater(() -> {
                                setPairsTraining(false);
                                setStatusMessage("Training completed successfully!");
                            });
                        } else {
                            String errorMsg = job.has("error") ? job.get("error").asText() : "Unknown error";
                            Platform.runLater(() -> {
                                setPairsTraining(false);
                                setStatusMessage("Training failed: " + errorMsg);
                            });
                        }
                        statusPollingScheduler.shutdown();
                    }
                }
            } catch (Exception e) {
                logger.error("Error polling training status", e);
            }
        }, 0, 5, TimeUnit.SECONDS);
    }

    /**
     * Create a new training pair with impaired and clear audio.
     *
     * @param impairedAudioFile File containing impaired speech
     * @param clearAudioFile File containing clear speech
     * @param prompt Text that was spoken
     */
    public void createTrainingPair(File impairedAudioFile, File clearAudioFile, String prompt) {
        UserProfile profile = selectedProfile.get();
        if (profile == null) {
            setStatusMessage("No profile selected");
            return;
        }

        String userId = profile.getId();
        setStatusMessage("Creating training pair...");

        executorService.submit(() -> {
            try {
                // Build multipart request
                String boundary = "WillSpeakBoundary" + System.currentTimeMillis();
                ByteArrayOutputStream requestBody = new ByteArrayOutputStream();

                // Add user_id part
                String userIdPartHeader =
                        "--" + boundary + "\r\n" +
                                "Content-Disposition: form-data; name=\"user_id\"\r\n\r\n";
                requestBody.write(userIdPartHeader.getBytes());
                requestBody.write(userId.getBytes());
                requestBody.write("\r\n".getBytes());

                // Add prompt part
                String promptPartHeader =
                        "--" + boundary + "\r\n" +
                                "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n";
                requestBody.write(promptPartHeader.getBytes());
                requestBody.write(prompt.getBytes());
                requestBody.write("\r\n".getBytes());

                // Add impaired audio file part
                String impairedPartHeader =
                        "--" + boundary + "\r\n" +
                                "Content-Disposition: form-data; name=\"impaired_audio\"; filename=\"" +
                                impairedAudioFile.getName() + "\"\r\n" +
                                "Content-Type: audio/wav\r\n\r\n";
                requestBody.write(impairedPartHeader.getBytes());
                requestBody.write(Files.readAllBytes(impairedAudioFile.toPath()));
                requestBody.write("\r\n".getBytes());

                // Add clear audio file part
                String clearPartHeader =
                        "--" + boundary + "\r\n" +
                                "Content-Disposition: form-data; name=\"clear_audio\"; filename=\"" +
                                clearAudioFile.getName() + "\"\r\n" +
                                "Content-Type: audio/wav\r\n\r\n";
                requestBody.write(clearPartHeader.getBytes());
                requestBody.write(Files.readAllBytes(clearAudioFile.toPath()));
                requestBody.write(("\r\n--" + boundary + "--\r\n").getBytes());

                // Create HTTP request
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(speechProcessingService.getServerUrl() + "/training-data/create-pair"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(requestBody.toByteArray()))
                        .build();

                // Send request
                HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    JsonNode responseJson = objectMapper.readTree(response.body());
                    String pairId = responseJson.get("id").asText();

                    Platform.runLater(() -> {
                        setStatusMessage("Training pair created successfully. ID: " + pairId);
                        // Refresh the list of training pairs
                        loadTrainingPairs();
                    });
                } else {
                    Platform.runLater(() -> {
                        setStatusMessage("Failed to create training pair: " + response.body());
                    });
                }
            } catch (Exception e) {
                logger.error("Error creating training pair", e);
                Platform.runLater(() -> {
                    setStatusMessage("Error creating training pair: " + e.getMessage());
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
     * Cancel any ongoing pair training.
     */
    public void cancelPairTraining() {
        if (!pairsTraining.get()) {
            return;
        }

        if (statusPollingScheduler != null && !statusPollingScheduler.isShutdown()) {
            statusPollingScheduler.shutdownNow();
            statusPollingScheduler = null;
        }

        setPairsTraining(false);
        setStatusMessage("Pair training canceled");
    }

    /**
     * Clean up resources.
     */
    public void shutdown() {
        if (audioCaptureService.isRecording()) {
            audioCaptureService.stopRecording();
        }

        if (statusPollingScheduler != null && !statusPollingScheduler.isShutdown()) {
            statusPollingScheduler.shutdownNow();
        }

        executorService.shutdown();
    }

    /**
     * Debug method to log all profiles.
     */
    public void logAllProfiles() {
        logger.info("=========== CURRENT PROFILES ===========");
        for (UserProfile profile : profiles) {
            logger.info("Profile: {} (ID: {})", profile.getName(), profile.getId());
        }
        logger.info("=======================================");
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

    public ObservableList<JsonNode> getTrainingPairs() {
        return trainingPairs;
    }

    public boolean isPairsTraining() {
        return pairsTraining.get();
    }

    public BooleanProperty pairsTrainingProperty() {
        return pairsTraining;
    }

    public void setPairsTraining(boolean pairsTraining) {
        this.pairsTraining.set(pairsTraining);
    }
}