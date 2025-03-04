package com.rednetty.willspeak;

import com.rednetty.willspeak.controller.MainController;
import com.rednetty.willspeak.controller.TrainingController;
import com.rednetty.willspeak.model.UserProfile;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javafx.util.StringConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;

/**
 * Main application class for WillSpeak speech enhancement system.
 */
public class WillSpeakApp extends Application {

    private static final Logger logger = LoggerFactory.getLogger(WillSpeakApp.class);
    private MainController mainController;
    private TrainingController trainingController;
    private Label statusLabel;
    private Label serverStatusLabel;
    private TextArea transcriptionArea;
    private ProgressIndicator processingIndicator;
    private ComboBox<UserProfile> profileComboBox;

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

        logger.info("Application UI initialized successfully");
    }

    private Tab createRealtimeTab() {
        Tab tab = new Tab("Real-time Assistant");
        tab.setClosable(false);

        // Main container
        BorderPane content = new BorderPane();
        content.setPadding(new Insets(15));

        // Controls pane
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
        userProfileCombo.setPrefWidth(Double.MAX_VALUE);
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

        // Transcription display area
        VBox transcriptionPane = new VBox(10);
        transcriptionPane.setPadding(new Insets(0, 0, 0, 15));

        Label transcriptionLabel = new Label("Transcription");
        transcriptionLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        transcriptionArea = new TextArea();
        transcriptionArea.setEditable(false);
        transcriptionArea.setWrapText(true);
        transcriptionArea.setPrefHeight(300);
        transcriptionArea.textProperty().bind(mainController.transcriptionTextProperty());

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
            // Get selected user profile
            UserProfile selectedProfile = userProfileCombo.getValue();
            String userId = selectedProfile != null ? selectedProfile.getId() : null;

            // Process with selected profile
            mainController.processRecordedAudio(userId);
            playButton.setDisable(false);
        });

        playButton.setOnAction(e -> {
            mainController.playProcessedAudio();
        });

        startRealTimeButton.setOnAction(e -> {
            // Get selected user profile
            UserProfile selectedProfile = userProfileCombo.getValue();
            String userId = selectedProfile != null ? selectedProfile.getId() : null;

            // Start real-time processing with selected profile
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

        BorderPane content = new BorderPane();
        content.setPadding(new Insets(15));

        VBox controlsPane = new VBox(15);
        controlsPane.setPadding(new Insets(10));

        Label titleLabel = new Label("Speech Model Training");
        titleLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 16px;");

        // User profile section
        Label profileLabel = new Label("User Profile");
        profileLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        // Profile selection
        profileComboBox = new ComboBox<>();
        profileComboBox.setPromptText("Select User Profile");
        profileComboBox.setItems(trainingController.getProfiles());
        profileComboBox.setPrefWidth(250);
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

        // New profile button
        Button newProfileButton = new Button("Create New Profile");
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
        trainingProgress.setPrefWidth(Double.MAX_VALUE);
        Label progressLabel = new Label("Ready to start training");

        // Current prompt display
        TextArea promptArea = new TextArea();
        promptArea.setEditable(false);
        promptArea.setWrapText(true);
        promptArea.setPrefHeight(80);
        promptArea.setPrefWidth(Double.MAX_VALUE);

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

        // Training instructions pane
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

        instructionsPane.getChildren().addAll(instructionsTitle, instructionsArea);

        // Bind controller to UI
        trainingController.selectedProfileProperty().bind(profileComboBox.valueProperty());
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

    private Tab createSettingsTab() {
        Tab tab = new Tab("Settings");
        tab.setClosable(false);

        BorderPane content = new BorderPane();
        content.setPadding(new Insets(15));

        VBox settingsPane = new VBox(15);
        settingsPane.setPadding(new Insets(10));

        Label titleLabel = new Label("Application Settings");
        titleLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 16px;");

        // Server connection settings
        Label serverLabel = new Label("Server Connection");
        serverLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        TextField serverUrlField = new TextField(DEFAULT_SERVER_URL);
        serverUrlField.setPrefWidth(300);

        Button connectButton = new Button("Test Connection");

        // Audio settings
        Label audioLabel = new Label("Audio Settings");
        audioLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        audioLabel.setPadding(new Insets(15, 0, 0, 0));

        ComboBox<String> inputDeviceCombo = new ComboBox<>();
        inputDeviceCombo.setPromptText("Select Input Device");
        inputDeviceCombo.getItems().addAll("Default Input", "Microphone", "Line In");
        inputDeviceCombo.setPrefWidth(300);

        ComboBox<String> outputDeviceCombo = new ComboBox<>();
        outputDeviceCombo.setPromptText("Select Output Device");
        outputDeviceCombo.getItems().addAll("Default Output", "Speakers", "Headphones");
        outputDeviceCombo.setPrefWidth(300);

        Label qualityLabel = new Label("Audio Quality:");
        ComboBox<String> qualityCombo = new ComboBox<>();
        qualityCombo.getItems().addAll("Low (8kHz)", "Medium (16kHz)", "High (44.1kHz)");
        qualityCombo.setValue("Medium (16kHz)");
        qualityCombo.setPrefWidth(300);

        // Profile management
        Label profileManagementLabel = new Label("Profile Management");
        profileManagementLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        profileManagementLabel.setPadding(new Insets(15, 0, 0, 0));

        Button syncProfilesButton = new Button("Synchronize Profiles with Server");
        syncProfilesButton.setPrefWidth(300);

        Button refreshProfilesButton = new Button("Refresh Local Profiles");
        refreshProfilesButton.setPrefWidth(300);

        // Save settings
        Button saveSettingsButton = new Button("Save Settings");
        saveSettingsButton.setPrefWidth(150);

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
                new HBox(10, saveSettingsButton)
        );

        // Button event handlers
        connectButton.setOnAction(e -> {
            mainController.checkServerConnection();
        });

        refreshProfilesButton.setOnAction(e -> {
            trainingController.loadProfiles();
        });

        syncProfilesButton.setOnAction(e -> {
            // TODO: Implement profile synchronization
            showAlert("Not Implemented", "Profile synchronization will be implemented in a future update.");
        });

        saveSettingsButton.setOnAction(e -> {
            // Save server URL if changed
            String newUrl = serverUrlField.getText();
            if (!newUrl.equals(DEFAULT_SERVER_URL)) {
                // TODO: Update controllers with new URL
                showAlert("Settings Saved", "Server URL updated. Please restart the application for changes to take effect.");
            }

            logger.info("Settings saved");
        });

        // Layout
        content.setLeft(settingsPane);

        tab.setContent(content);
        return tab;
    }

    /**
     * Display a dialog to create a new user profile.
     */
    private void showCreateProfileDialog() {
        // Create the dialog
        Dialog<UserProfile> dialog = new Dialog<>();
        dialog.setTitle("Create New Profile");
        dialog.setHeaderText("Enter profile information");

        // Set the button types
        ButtonType createButtonType = new ButtonType("Create", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(createButtonType, ButtonType.CANCEL);

        // Create the form grid
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

        // Enable/Disable create button depending on whether a name was entered
        Button createButton = (Button) dialog.getDialogPane().lookupButton(createButtonType);
        createButton.setDisable(true);

        // Validation
        nameField.textProperty().addListener((observable, oldValue, newValue) -> {
            createButton.setDisable(newValue.trim().isEmpty());
        });

        // Convert the result to a profile when the create button is clicked
        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == createButtonType) {
                String name = nameField.getText();
                String description = descriptionField.getText();

                // Create profile locally and on server
                trainingController.createServerProfile(name, description);
                return new UserProfile(name);
            }
            return null;
        });

        Optional<UserProfile> result = dialog.showAndWait();

        // No need to handle the result directly as the controller already does this
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
        // Clean up resources
        if (mainController != null) {
            mainController.shutdown();
        }
        if (trainingController != null) {
            trainingController.shutdown();
        }
        logger.info("Application shutting down");
    }
}