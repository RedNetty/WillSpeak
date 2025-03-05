package com.rednetty.willspeak;

import com.fasterxml.jackson.databind.JsonNode;
import com.rednetty.willspeak.controller.MainController;
import com.rednetty.willspeak.controller.TrainingController;
import com.rednetty.willspeak.controller.TrainingDataController;
import com.rednetty.willspeak.model.UserProfile;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.util.StringConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Main application class for WillSpeak speech enhancement system.
 */
public class WillSpeakApp extends Application {

    private static final Logger logger = LoggerFactory.getLogger(WillSpeakApp.class);
    private MainController mainController;
    private TrainingController trainingController;
    private TrainingDataController trainingDataController;
    private Label statusLabel;
    private Label serverStatusLabel;
    private TextArea transcriptionArea;
    private ProgressIndicator processingIndicator;
    private ComboBox<UserProfile> profileComboBox; // For Prompt Training tab
    private ListView<JsonNode> pairsListView;

    // File tracking for training pairs
    private File lastRecordedImpairedFile;
    private File lastRecordedClearFile;

    // Recording state tracking
    private Button activeRecordButton = null;
    private String currentRecordingType = null;
    private Path tempAudioPath = null;
    private boolean isCurrentlyRecording = false;

    // Server URL configuration - could be moved to properties file
    private static final String DEFAULT_SERVER_URL = "http://localhost:8000";

    public static void main(String[] args) {
        logger.info("Starting WillSpeak application");
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        logger.info("Initializing UI components");

        // Initialize controllers
        mainController = new MainController(DEFAULT_SERVER_URL);
        trainingController = new TrainingController(DEFAULT_SERVER_URL);
        trainingDataController = new TrainingDataController(DEFAULT_SERVER_URL);

        // Create root layout
        BorderPane root = new BorderPane();

        // Create tab pane for main sections
        TabPane tabPane = new TabPane();

        // Create tabs
        Tab realtimeTab = createRealtimeTab();
        Tab trainingTab = createTrainingTab();
        Tab settingsTab = createSettingsTab();

        tabPane.getTabs().addAll(realtimeTab, trainingTab, settingsTab);

        // Add components to root
        root.setCenter(tabPane);

        // Create status bar
        statusLabel = new Label("Ready");
        processingIndicator = new ProgressIndicator(-1);
        processingIndicator.setVisible(false);
        processingIndicator.setPrefSize(20, 20);

        serverStatusLabel = new Label("Server Status: Checking...");
        serverStatusLabel.setPadding(new Insets(0, 10, 0, 0));

        HBox statusComponents = new HBox(10);
        statusComponents.setAlignment(Pos.CENTER_LEFT);
        statusComponents.getChildren().addAll(processingIndicator, statusLabel);

        BorderPane statusBar = new BorderPane();
        statusBar.setLeft(statusComponents);
        statusBar.setRight(serverStatusLabel);
        statusBar.setPadding(new Insets(5));
        statusBar.setStyle("-fx-background-color: #f0f0f0;");
        root.setBottom(statusBar);

        // Create scene
        Scene scene = new Scene(root, 800, 600);

        // Configure stage
        primaryStage.setTitle("WillSpeak - Speech Enhancement System");
        primaryStage.setScene(scene);
        primaryStage.show();

        // Bind UI properties to main controller
        statusLabel.textProperty().bind(mainController.statusMessageProperty());
        processingIndicator.visibleProperty().bind(mainController.processingProperty());

        // Bind server status
        mainController.serverConnectedProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal) {
                serverStatusLabel.setText("Server Status: Connected");
                serverStatusLabel.setTextFill(Color.GREEN);
            } else {
                serverStatusLabel.setText("Server Status: Disconnected");
                serverStatusLabel.setTextFill(Color.RED);
            }
        });

        // Check server connection
        mainController.checkServerConnection();

        // Load user profiles
        trainingController.loadProfiles();
        trainingDataController.loadProfiles();

        logger.info("Application UI initialized successfully");
    }

    private Tab createRealtimeTab() {
        Tab tab = new Tab("Real-time Assistant");
        tab.setClosable(false);

        // Main container
        BorderPane content = new BorderPane();
        content.setPadding(new Insets(15));

        // Controls pane (Left Side)
        VBox controlsPane = new VBox(10);
        controlsPane.setPadding(new Insets(0, 0, 0, 0));
        controlsPane.setMinWidth(200);

        // Title
        Label titleLabel = new Label("Speech Controls");
        titleLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        controlsPane.getChildren().add(titleLabel);

        // User profile selection
        Label profileLabel = new Label("User Profile:");
        ComboBox<UserProfile> userProfileCombo = new ComboBox<>();
        userProfileCombo.setPromptText("Select User Profile");
        userProfileCombo.setItems(trainingController.getProfiles());
        userProfileCombo.setMaxWidth(Double.MAX_VALUE);
        userProfileCombo.setConverter(new StringConverter<UserProfile>() {
            @Override
            public String toString(UserProfile profile) {
                return profile == null ? "Default" : profile.getName();
            }

            @Override
            public UserProfile fromString(String string) {
                return null;
            }
        });

        // Buttons for recording
        Button recordButton = new Button("Start Recording");
        recordButton.setMaxWidth(Double.MAX_VALUE);
        Button stopButton = new Button("Stop");
        stopButton.setMaxWidth(Double.MAX_VALUE);
        stopButton.setDisable(true);

        // Buttons for processing
        Button enhanceButton = new Button("Enhance Speech");
        enhanceButton.setMaxWidth(Double.MAX_VALUE);
        enhanceButton.setDisable(true);

        Button playButton = new Button("Play Processed Audio");
        playButton.setMaxWidth(Double.MAX_VALUE);
        playButton.setDisable(true);

        // Real-time processing controls
        Label realtimeLabel = new Label("Real-time Processing");
        realtimeLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        realtimeLabel.setPadding(new Insets(10, 0, 0, 0));

        Button startRealTimeButton = new Button("Start Real-time Mode");
        startRealTimeButton.setMaxWidth(Double.MAX_VALUE);
        Button stopRealTimeButton = new Button("Stop Real-time Mode");
        stopRealTimeButton.setMaxWidth(Double.MAX_VALUE);
        stopRealTimeButton.setDisable(true);

        // Add all controls
        controlsPane.getChildren().addAll(
                profileLabel,
                userProfileCombo,
                recordButton,
                stopButton,
                enhanceButton,
                playButton,
                realtimeLabel,
                startRealTimeButton,
                stopRealTimeButton
        );

        // Transcription display area (Center)
        VBox transcriptionPane = new VBox(10);
        transcriptionPane.setPadding(new Insets(0, 0, 0, 15));

        Label transcriptionLabel = new Label("Transcription");
        transcriptionLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        transcriptionArea = new TextArea();
        transcriptionArea.setEditable(false);
        transcriptionArea.setWrapText(true);
        transcriptionArea.textProperty().bind(mainController.transcriptionTextProperty());
        VBox.setVgrow(transcriptionArea, Priority.ALWAYS); // Allow vertical growth

        transcriptionPane.getChildren().addAll(transcriptionLabel, transcriptionArea);

        // Button event handlers
        recordButton.setOnAction(e -> {
            mainController.startRecording();
            recordButton.setDisable(true);
            stopButton.setDisable(false);
            enhanceButton.setDisable(true);
            playButton.setDisable(true);
            startRealTimeButton.setDisable(true);
        });

        stopButton.setOnAction(e -> {
            mainController.stopRecording();
            recordButton.setDisable(false);
            stopButton.setDisable(true);
            enhanceButton.setDisable(false);
            startRealTimeButton.setDisable(false);
        });

        enhanceButton.setOnAction(e -> {
            UserProfile selectedProfile = userProfileCombo.getValue();
            String userId = selectedProfile != null ? selectedProfile.getId() : null;
            mainController.processRecordedAudio(userId);
            playButton.setDisable(false);
        });

        playButton.setOnAction(e -> mainController.playProcessedAudio());

        startRealTimeButton.setOnAction(e -> {
            UserProfile selectedProfile = userProfileCombo.getValue();
            String userId = selectedProfile != null ? selectedProfile.getId() : null;
            mainController.startRealTimeProcessing(userId);
            recordButton.setDisable(true);
            stopButton.setDisable(true);
            enhanceButton.setDisable(true);
            playButton.setDisable(true);
            startRealTimeButton.setDisable(true);
            stopRealTimeButton.setDisable(false);
        });

        stopRealTimeButton.setOnAction(e -> {
            mainController.stopRealTimeProcessing();
            recordButton.setDisable(false);
            stopRealTimeButton.setDisable(true);
            startRealTimeButton.setDisable(false);
        });

        // Bind button states to controller properties
        mainController.recordingProperty().addListener((obs, oldVal, newVal) -> {
            if (!newVal) {
                Platform.runLater(() -> {
                    recordButton.setDisable(false);
                    stopButton.setDisable(true);
                    enhanceButton.setDisable(false);
                });
            }
        });

        // Add components to layout
        content.setLeft(controlsPane);
        content.setCenter(transcriptionPane);

        tab.setContent(content);
        return tab;
    }

    private Tab createTrainingTab() {
        Tab tab = new Tab("Training");
        tab.setClosable(false);

        // Create a root TabPane for the training tab to hold different training methods
        TabPane trainingTabPane = new TabPane();
        trainingTabPane.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);

        // Create tabs for different training methods
        Tab promptTrainingTab = createPromptTrainingTab();
        Tab pairsTrainingTab = createPairsTrainingTab();

        trainingTabPane.getTabs().addAll(promptTrainingTab, pairsTrainingTab);

        // Set the TabPane as the content of the main training tab
        tab.setContent(trainingTabPane);

        return tab;
    }

    private Tab createPromptTrainingTab() {
        Tab tab = new Tab("Prompt Training");

        BorderPane content = new BorderPane();
        content.setPadding(new Insets(15));

        // Controls pane (Left Side)
        VBox controlsPane = new VBox(15);
        controlsPane.setPadding(new Insets(10));
        controlsPane.setMinWidth(200);

        Label titleLabel = new Label("Prompt-based Training");
        titleLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 16px;");

        // User profile section
        Label profileLabel = new Label("User Profile");
        profileLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        profileComboBox = new ComboBox<>();
        profileComboBox.setPromptText("Select User Profile");
        profileComboBox.setItems(trainingController.getProfiles());
        profileComboBox.setMaxWidth(Double.MAX_VALUE);
        profileComboBox.setConverter(new StringConverter<UserProfile>() {
            @Override
            public String toString(UserProfile profile) {
                return profile == null ? "" : profile.getName();
            }

            @Override
            public UserProfile fromString(String string) {
                return null;
            }
        });

        Button newProfileButton = new Button("Create New Profile");
        newProfileButton.setMaxWidth(Double.MAX_VALUE);
        newProfileButton.setOnAction(e -> showCreateProfileDialog());

        // Training controls
        Label trainingLabel = new Label("Training Session");
        trainingLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        trainingLabel.setPadding(new Insets(15, 0, 0, 0));

        Button startTrainingButton = new Button("Start Training Session");
        startTrainingButton.setMaxWidth(Double.MAX_VALUE);

        Button recordPromptButton = new Button("Record Prompt");
        recordPromptButton.setMaxWidth(Double.MAX_VALUE);
        recordPromptButton.setDisable(true);

        Button stopRecordingButton = new Button("Stop Recording");
        stopRecordingButton.setMaxWidth(Double.MAX_VALUE);
        stopRecordingButton.setDisable(true);

        Button reviewResultsButton = new Button("Review Training Results");
        reviewResultsButton.setMaxWidth(Double.MAX_VALUE);
        reviewResultsButton.setDisable(true);

        // Training progress
        ProgressBar trainingProgress = new ProgressBar(0);
        trainingProgress.setMaxWidth(Double.MAX_VALUE);
        Label progressLabel = new Label("Ready to start training");

        // Current prompt display
        TextArea promptArea = new TextArea();
        promptArea.setEditable(false);
        promptArea.setWrapText(true);
        promptArea.setPrefHeight(80);
        promptArea.setMaxWidth(Double.MAX_VALUE);

        // Add all controls
        controlsPane.getChildren().addAll(
                titleLabel,
                profileLabel,
                profileComboBox,
                newProfileButton,
                trainingLabel,
                startTrainingButton,
                new Label("Current Prompt:"),
                promptArea,
                recordPromptButton,
                stopRecordingButton,
                reviewResultsButton,
                new Label("Progress:"),
                trainingProgress,
                progressLabel
        );

        // Instructions pane (Center)
        VBox instructionsPane = new VBox(10);
        instructionsPane.setPadding(new Insets(10, 10, 10, 20));

        Label instructionsTitle = new Label("Training Instructions");
        instructionsTitle.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        TextArea instructionsArea = new TextArea(
                "Training helps WillSpeak learn your unique speech patterns.\n\n" +
                        "During a training session, you will:\n" +
                        "1. Read a series of provided sentences\n" +
                        "2. The system will record and analyze your speech\n" +
                        "3. A personalized model will be created just for you\n\n" +
                        "The more you train, the better the system becomes at enhancing your speech."
        );
        instructionsArea.setEditable(false);
        instructionsArea.setWrapText(true);
        instructionsArea.setPrefRowCount(10);
        VBox.setVgrow(instructionsArea, Priority.ALWAYS);

        instructionsPane.getChildren().addAll(instructionsTitle, instructionsArea);

        // Bind controller to UI
        promptArea.textProperty().bind(trainingController.statusMessageProperty());
        trainingProgress.progressProperty().bind(trainingController.trainingProgressProperty());

        // Bind training state to UI
        trainingController.trainingProperty().addListener((obs, oldVal, newVal) -> {
            Platform.runLater(() -> {
                startTrainingButton.setDisable(newVal);
                recordPromptButton.setDisable(!newVal);
                profileComboBox.setDisable(newVal);
                newProfileButton.setDisable(newVal);
                if (!newVal) {
                    stopRecordingButton.setDisable(true);
                    progressLabel.setText("Training complete");
                } else {
                    progressLabel.setText("Training in progress...");
                }
            });
        });

        // Add listeners for profile selection synchronization
        profileComboBox.valueProperty().addListener((obs, oldVal, newVal) -> {
            trainingController.setSelectedProfile(newVal);
        });

        trainingController.selectedProfileProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != profileComboBox.getValue()) {
                profileComboBox.setValue(newVal);
            }
        });

        // Button event handlers
        startTrainingButton.setOnAction(e -> {
            UserProfile selectedProfile = profileComboBox.getValue();
            if (selectedProfile == null) {
                showAlert("No Profile Selected", "Please select or create a user profile first.");
                return;
            }
            trainingController.startTrainingSession();
        });

        recordPromptButton.setOnAction(e -> {
            trainingController.recordTrainingAudio();
            recordPromptButton.setDisable(true);
            stopRecordingButton.setDisable(false);
        });

        stopRecordingButton.setOnAction(e -> {
            trainingController.stopRecordingAndSave();
            recordPromptButton.setDisable(false);
            stopRecordingButton.setDisable(true);
        });

        // Layout
        content.setLeft(controlsPane);
        content.setCenter(instructionsPane);

        tab.setContent(content);
        return tab;
    }


    // Create the Training Pairs tab
    private Tab createPairsTrainingTab() {
        Tab tab = new Tab("Training Pairs");

        BorderPane content = new BorderPane();
        content.setPadding(new Insets(15));

        // Left panel - controls
        VBox controlsPane = new VBox(10);
        controlsPane.setPadding(new Insets(10));
        controlsPane.setMinWidth(200);

        Label titleLabel = new Label("Training Pair Management");
        titleLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 16px;");

        // User profile section
        Label profileLabel = new Label("User Profile");
        profileLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        ComboBox<UserProfile> userProfileCombo = new ComboBox<>();
        userProfileCombo.setPromptText("Select User Profile");
        userProfileCombo.setItems(trainingDataController.getProfiles());
        userProfileCombo.setMaxWidth(Double.MAX_VALUE);
        userProfileCombo.setConverter(new StringConverter<UserProfile>() {
            @Override
            public String toString(UserProfile profile) {
                return profile == null ? "" : profile.getName();
            }

            @Override
            public UserProfile fromString(String string) {
                return null;
            }
        });

        Button loadProfileButton = new Button("Load Profile Pairs");
        loadProfileButton.setMaxWidth(Double.MAX_VALUE);

        // Training pairs management
        Label pairsLabel = new Label("Create Training Pair");
        pairsLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        pairsLabel.setPadding(new Insets(15, 0, 0, 0));

        TextField promptField = new TextField();
        promptField.setPromptText("Enter prompt text");
        promptField.setMaxWidth(Double.MAX_VALUE);

        Button recordImpairedButton = new Button("Record Impaired Speech");
        recordImpairedButton.setMaxWidth(Double.MAX_VALUE);

        Button recordClearButton = new Button("Record Clear Speech");
        recordClearButton.setMaxWidth(Double.MAX_VALUE);

        Button stopRecordingButton = new Button("Stop Recording");
        stopRecordingButton.setMaxWidth(Double.MAX_VALUE);
        stopRecordingButton.setDisable(true);

        Button importImpairedButton = new Button("Import Impaired Speech");
        importImpairedButton.setMaxWidth(Double.MAX_VALUE);

        Button importClearButton = new Button("Import Clear Speech");
        importClearButton.setMaxWidth(Double.MAX_VALUE);

        Button createPairButton = new Button("Create Training Pair");
        createPairButton.setMaxWidth(Double.MAX_VALUE);
        createPairButton.setDisable(true);

        // Add listener to prompt field to update button state
        promptField.textProperty().addListener((obs, oldVal, newVal) -> {
            updateCreatePairButton(createPairButton, promptField);
        });

        Label trainingLabel = new Label("Model Training");
        trainingLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        trainingLabel.setPadding(new Insets(15, 0, 0, 0));

        Button startTrainingButton = new Button("Train from Pairs");
        startTrainingButton.setMaxWidth(Double.MAX_VALUE);

        Button cancelTrainingButton = new Button("Cancel Training");
        cancelTrainingButton.setMaxWidth(Double.MAX_VALUE);
        cancelTrainingButton.setDisable(true);

        ProgressIndicator trainingIndicator = new ProgressIndicator();
        trainingIndicator.setVisible(false);
        trainingIndicator.setMaxWidth(Double.MAX_VALUE);

        // Add all controls
        controlsPane.getChildren().addAll(
                titleLabel,
                profileLabel,
                userProfileCombo,
                loadProfileButton,
                pairsLabel,
                new Label("Prompt:"),
                promptField,
                recordImpairedButton,
                recordClearButton,
                stopRecordingButton,
                importImpairedButton,
                importClearButton,
                createPairButton,
                trainingLabel,
                startTrainingButton,
                cancelTrainingButton,
                trainingIndicator
        );

        // Right panel - training pairs list and details
        VBox pairsPane = new VBox(10);
        pairsPane.setPadding(new Insets(10));

        Label pairsListLabel = new Label("Available Training Pairs");
        pairsListLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        pairsListView = new ListView<>();
        pairsListView.setPrefHeight(300);
        VBox.setVgrow(pairsListView, Priority.ALWAYS);
        pairsListView.setCellFactory(lv -> new ListCell<>() {
            @Override
            protected void updateItem(JsonNode item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                } else {
                    String prompt = item.has("prompt") ? item.get("prompt").asText() : "No prompt";
                    String created = item.has("created") ? item.get("created").asText().substring(0, 16) : "";
                    setText(prompt + " [" + created + "]");
                }
            }
        });

        Label detailsLabel = new Label("Selected Pair Details");
        detailsLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        TextArea pairDetailsArea = new TextArea();
        pairDetailsArea.setEditable(false);
        pairDetailsArea.setWrapText(true);
        pairDetailsArea.setPrefHeight(150);

        Button playImpairedButton = new Button("Play Impaired Speech");
        playImpairedButton.setMaxWidth(Double.MAX_VALUE);
        playImpairedButton.setDisable(true);

        Button playClearButton = new Button("Play Clear Speech");
        playClearButton.setMaxWidth(Double.MAX_VALUE);
        playClearButton.setDisable(true);

        pairsPane.getChildren().addAll(
                pairsListLabel,
                pairsListView,
                detailsLabel,
                pairDetailsArea,
                new HBox(10, playImpairedButton, playClearButton)
        );

        // Bind selected pair to details view
        pairsListView.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                StringBuilder details = new StringBuilder();
                details.append("ID: ").append(newVal.get("id").asText()).append("\n");
                details.append("Prompt: ").append(newVal.get("prompt").asText()).append("\n");
                details.append("Created: ").append(newVal.get("created").asText()).append("\n");
                if (newVal.has("notes") && !newVal.get("notes").isNull()) {
                    details.append("Notes: ").append(newVal.get("notes").asText()).append("\n");
                }
                pairDetailsArea.setText(details.toString());

                playImpairedButton.setDisable(false);
                playClearButton.setDisable(false);
            } else {
                pairDetailsArea.setText("");
                playImpairedButton.setDisable(true);
                playClearButton.setDisable(true);
            }
        });

        // Add listeners for profile selection synchronization
        userProfileCombo.valueProperty().addListener((obs, oldVal, newVal) -> {
            trainingDataController.setSelectedProfile(newVal);
        });

        trainingDataController.selectedProfileProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != userProfileCombo.getValue()) {
                userProfileCombo.setValue(newVal);
            }
        });

        // Set up button handlers
        loadProfileButton.setOnAction(e -> {
            trainingDataController.loadUserTrainingPairs();
            pairsListView.setItems(trainingDataController.getTrainingPairs());
        });

        // Record impaired speech button
        recordImpairedButton.setOnAction(e -> {
            if (isCurrentlyRecording) {
                // Don't allow starting a new recording while already recording
                return;
            }

            // Set up recording state
            isCurrentlyRecording = true;
            activeRecordButton = recordImpairedButton;
            currentRecordingType = "Impaired";

            // Get prompt text
            String prompt = promptField.getText();
            if (prompt.isEmpty()) {
                prompt = "Default prompt";
            }

            // Update UI first
            recordImpairedButton.setText("Recording Impaired...");
            stopRecordingButton.setDisable(false);
            recordClearButton.setDisable(true);
            importImpairedButton.setDisable(true);
            importClearButton.setDisable(true);
            createPairButton.setDisable(true);

            // Start recording
            trainingDataController.startRecording("impaired", prompt);
        });

        // Record clear speech button
        recordClearButton.setOnAction(e -> {
            if (isCurrentlyRecording) {
                // Don't allow starting a new recording while already recording
                return;
            }

            // Set up recording state
            isCurrentlyRecording = true;
            activeRecordButton = recordClearButton;
            currentRecordingType = "Clear";

            // Get prompt text
            String prompt = promptField.getText();
            if (prompt.isEmpty()) {
                prompt = "Default prompt";
            }

            // Update UI first
            recordClearButton.setText("Recording Clear...");
            stopRecordingButton.setDisable(false);
            recordImpairedButton.setDisable(true);
            importImpairedButton.setDisable(true);
            importClearButton.setDisable(true);
            createPairButton.setDisable(true);

            // Start recording
            trainingDataController.startRecording("clear", prompt);
        });

        // Stop recording button
        stopRecordingButton.setOnAction(e -> {
            if (isCurrentlyRecording) {
                // Stop the recording
                String filePath = trainingDataController.stopRecording();
                isCurrentlyRecording = false;

                if (filePath != null) {
                    // Update UI based on recording type
                    if (activeRecordButton != null) {
                        if ("Impaired".equals(currentRecordingType)) {
                            activeRecordButton.setText("Impaired Speech Recorded ✓");
                            lastRecordedImpairedFile = new File(filePath);
                        } else {
                            activeRecordButton.setText("Clear Speech Recorded ✓");
                            lastRecordedClearFile = new File(filePath);
                        }
                    }
                }

                // Reset button states
                stopRecordingButton.setDisable(true);
                recordImpairedButton.setDisable(false);
                recordClearButton.setDisable(false);
                importImpairedButton.setDisable(false);
                importClearButton.setDisable(false);

                // Update create pair button state
                updateCreatePairButton(createPairButton, promptField);
            }
        });

        // Import impaired speech button
        importImpairedButton.setOnAction(e -> {
            if (isCurrentlyRecording) {
                return; // Don't allow importing while recording
            }

            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Import Impaired Speech File");
            fileChooser.getExtensionFilters().add(
                    new FileChooser.ExtensionFilter("WAV Files", "*.wav")
            );
            File selectedFile = fileChooser.showOpenDialog(null);
            if (selectedFile != null) {
                lastRecordedImpairedFile = selectedFile;
                recordImpairedButton.setText("Impaired: " + selectedFile.getName());
                updateCreatePairButton(createPairButton, promptField);
            }
        });

        // Import clear speech button
        importClearButton.setOnAction(e -> {
            if (isCurrentlyRecording) {
                return; // Don't allow importing while recording
            }

            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Import Clear Speech File");
            fileChooser.getExtensionFilters().add(
                    new FileChooser.ExtensionFilter("WAV Files", "*.wav")
            );
            File selectedFile = fileChooser.showOpenDialog(null);
            if (selectedFile != null) {
                lastRecordedClearFile = selectedFile;
                recordClearButton.setText("Clear: " + selectedFile.getName());
                updateCreatePairButton(createPairButton, promptField);
            }
        });

        // Create training pair button
        createPairButton.setOnAction(e -> {
            if (isCurrentlyRecording) {
                return; // Don't allow creating pairs while recording
            }

            if (lastRecordedImpairedFile != null && lastRecordedClearFile != null && !promptField.getText().isEmpty()) {
                // Get the profile and create the training pair
                UserProfile profile = userProfileCombo.getValue();
                if (profile == null) {
                    showAlert("No Profile Selected", "Please select a user profile first.");
                    return;
                }

                // Update controller with file references
                trainingDataController.setImpairedRecording(lastRecordedImpairedFile);
                trainingDataController.setClearRecording(lastRecordedClearFile);

                // Create the training pair
                trainingDataController.createTrainingPair(promptField.getText(), null);

                // Reset form after submission
                lastRecordedImpairedFile = null;
                lastRecordedClearFile = null;
                promptField.clear();
                recordImpairedButton.setText("Record Impaired Speech");
                recordClearButton.setText("Record Clear Speech");
                createPairButton.setDisable(true);
            }
        });

        // Training buttons
        startTrainingButton.setOnAction(e -> {
            if (isCurrentlyRecording) {
                return; // Don't allow training while recording
            }

            if (pairsListView.getItems().isEmpty()) {
                showAlert("No Training Pairs", "Please create or load training pairs first.");
                return;
            }
            trainingDataController.trainFromPairs();
        });

        cancelTrainingButton.setOnAction(e -> {
            // We need to implement this in the TrainingDataController
            trainingDataController.setTrainingInProgress(false);
        });

        // Bind recording state to UI
        trainingDataController.recordingProperty().addListener((obs, oldVal, newVal) -> {
            // If recording stops unexpectedly
            if (!newVal && isCurrentlyRecording) {
                Platform.runLater(() -> {
                    isCurrentlyRecording = false;
                    stopRecordingButton.setDisable(true);
                    recordImpairedButton.setDisable(false);
                    recordClearButton.setDisable(false);
                    importImpairedButton.setDisable(false);
                    importClearButton.setDisable(false);

                    if (activeRecordButton != null) {
                        activeRecordButton.setText(currentRecordingType + " Speech");
                    }
                });
            }
        });

        // Bind training state
        trainingDataController.trainingInProgressProperty().addListener((obs, oldVal, newVal) -> {
            Platform.runLater(() -> {
                trainingIndicator.setVisible(newVal);
                startTrainingButton.setDisable(newVal);
                cancelTrainingButton.setDisable(!newVal);
                userProfileCombo.setDisable(newVal);
                loadProfileButton.setDisable(newVal);
            });
        });

        // Audio playback buttons
        playImpairedButton.setOnAction(e -> {
            if (isCurrentlyRecording) {
                return; // Don't allow playback while recording
            }

            JsonNode selectedPair = pairsListView.getSelectionModel().getSelectedItem();
            if (selectedPair != null) {
                trainingDataController.playTrainingPairAudio(selectedPair, "impaired");
            }
        });

        playClearButton.setOnAction(e -> {
            if (isCurrentlyRecording) {
                return; // Don't allow playback while recording
            }

            JsonNode selectedPair = pairsListView.getSelectionModel().getSelectedItem();
            if (selectedPair != null) {
                trainingDataController.playTrainingPairAudio(selectedPair, "clear");
            }
        });

        // Layout
        content.setLeft(controlsPane);
        content.setCenter(pairsPane);

        tab.setContent(content);
        return tab;
    }
    private Tab createSettingsTab() {
        Tab tab = new Tab("Settings");
        tab.setClosable(false);

        BorderPane content = new BorderPane();
        content.setPadding(new Insets(15));

        // Settings pane (Center)
        VBox settingsPane = new VBox(15);
        settingsPane.setPadding(new Insets(10));

        Label titleLabel = new Label("Application Settings");
        titleLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 16px;");

        // Server connection settings
        Label serverLabel = new Label("Server Connection");
        serverLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        TextField serverUrlField = new TextField(DEFAULT_SERVER_URL);
        serverUrlField.setMaxWidth(Double.MAX_VALUE);

        Button connectButton = new Button("Test Connection");
        connectButton.setMaxWidth(Double.MAX_VALUE);

        // Audio settings
        Label audioLabel = new Label("Audio Settings");
        audioLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        audioLabel.setPadding(new Insets(15, 0, 0, 0));

        ComboBox<String> inputDeviceCombo = new ComboBox<>();
        inputDeviceCombo.setPromptText("Select Input Device");
        inputDeviceCombo.getItems().addAll("Default Input", "Microphone", "Line In");
        inputDeviceCombo.setMaxWidth(Double.MAX_VALUE);

        ComboBox<String> outputDeviceCombo = new ComboBox<>();
        outputDeviceCombo.setPromptText("Select Output Device");
        outputDeviceCombo.getItems().addAll("Default Output", "Speakers", "Headphones");
        outputDeviceCombo.setMaxWidth(Double.MAX_VALUE);

        Label qualityLabel = new Label("Audio Quality:");
        ComboBox<String> qualityCombo = new ComboBox<>();
        qualityCombo.getItems().addAll("Low (8kHz)", "Medium (16kHz)", "High (44.1kHz)");
        qualityCombo.setValue("Medium (16kHz)");
        qualityCombo.setMaxWidth(Double.MAX_VALUE);

        // Profile management
        Label profileManagementLabel = new Label("Profile Management");
        profileManagementLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        profileManagementLabel.setPadding(new Insets(15, 0, 0, 0));

        Button syncProfilesButton = new Button("Synchronize Profiles with Server");
        syncProfilesButton.setMaxWidth(Double.MAX_VALUE);

        Button refreshProfilesButton = new Button("Refresh Local Profiles");
        refreshProfilesButton.setMaxWidth(Double.MAX_VALUE);

        // Save settings
        Button saveSettingsButton = new Button("Save Settings");
        saveSettingsButton.setMaxWidth(Double.MAX_VALUE);

        // Add all controls
        settingsPane.getChildren().addAll(
                titleLabel,
                serverLabel,
                new Label("Server URL:"),
                serverUrlField,
                connectButton,
                audioLabel,
                new Label("Input Device:"),
                inputDeviceCombo,
                new Label("Output Device:"),
                outputDeviceCombo,
                qualityLabel,
                qualityCombo,
                profileManagementLabel,
                syncProfilesButton,
                refreshProfilesButton,
                saveSettingsButton
        );

        // Button event handlers
        connectButton.setOnAction(e -> mainController.checkServerConnection());

        refreshProfilesButton.setOnAction(e -> {
            trainingController.loadProfiles();
            trainingDataController.loadProfiles();
        });

        syncProfilesButton.setOnAction(e -> {
            showAlert("Not Implemented", "Profile synchronization will be implemented in a future update.");
        });

        saveSettingsButton.setOnAction(e -> {
            String newUrl = serverUrlField.getText();
            if (!newUrl.equals(DEFAULT_SERVER_URL)) {
                showAlert("Settings Saved", "Server URL updated. Please restart the application for changes to take effect.");
            }
            logger.info("Settings saved");
        });

        // Layout
        content.setCenter(settingsPane);

        tab.setContent(content);
        return tab;
    }

    /**
     * Display a dialog to create a new user profile.
     */
    private void showCreateProfileDialog() {
        Dialog<UserProfile> dialog = new Dialog<>();
        dialog.setTitle("Create New Profile");
        dialog.setHeaderText("Enter profile information");

        ButtonType createButtonType = new ButtonType("Create", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(createButtonType, ButtonType.CANCEL);

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(20, 150, 10, 10));

        TextField nameField = new TextField();
        nameField.setPromptText("Name");
        TextField descriptionField = new TextField();
        descriptionField.setPromptText("Description (optional)");

        grid.add(new Label("Name:"), 0, 0);
        grid.add(nameField, 1, 0);
        grid.add(new Label("Description:"), 0, 1);
        grid.add(descriptionField, 1, 1);

        dialog.getDialogPane().setContent(grid);

        Button createButton = (Button) dialog.getDialogPane().lookupButton(createButtonType);
        createButton.setDisable(true);

        nameField.textProperty().addListener((observable, oldValue, newValue) -> {
            createButton.setDisable(newValue.trim().isEmpty());
        });

        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == createButtonType) {
                String name = nameField.getText();
                String description = descriptionField.getText();

                // Create both in training controller and training data controller
                trainingController.createServerProfile(name, description);

                return new UserProfile(name);
            }
            return null;
        });

        dialog.showAndWait();
    }

    /**
     * Update create pair button state based on form completion
     */
    private void updateCreatePairButton(Button createPairButton, TextField promptField) {
        boolean hasImpaired = lastRecordedImpairedFile != null;
        boolean hasClear = lastRecordedClearFile != null;
        boolean hasPrompt = promptField != null && !promptField.getText().isEmpty();

        createPairButton.setDisable(!(hasImpaired && hasClear && hasPrompt));
    }

    /**
     * Display an alert dialog.
     */
    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    @Override
    public void stop() {
        if (mainController != null) {
            mainController.shutdown();
        }
        if (trainingController != null) {
            trainingController.shutdown();
        }
        if (trainingDataController != null) {
            trainingDataController.shutdown();
        }
        logger.info("Application shutting down");
    }
}