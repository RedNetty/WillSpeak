package com.rednetty.willspeak.service;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Service for processing speech through the WillSpeak server.
 */
public class SpeechProcessingService {
    private static final Logger logger = LoggerFactory.getLogger(SpeechProcessingService.class);

    private final String serverUrl;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private WebSocket webSocket;
    private Consumer<byte[]> audioDataConsumer;

    public SpeechProcessingService(String serverUrl) {
        this.serverUrl = serverUrl;
        this.httpClient = HttpClient.newBuilder().build();
        this.objectMapper = new ObjectMapper();
        logger.info("SpeechProcessingService initialized with server URL: {}", serverUrl);
    }

    /**
     * Process an audio file through the server via HTTP.
     *
     * @param audioBytes the audio data to process
     * @return a JsonNode containing the server response
     * @throws IOException if an I/O error occurs
     * @throws InterruptedException if the operation is interrupted
     */
    public JsonNode processAudio(byte[] audioBytes) throws IOException, InterruptedException {
        return processAudio(audioBytes, null);
    }

    /**
     * Process an audio file through the server with a specific user profile.
     *
     * @param audioBytes the audio data to process
     * @param userId the user ID to process with (may be null)
     * @return a JsonNode containing the server response
     * @throws IOException if an I/O error occurs
     * @throws InterruptedException if the operation is interrupted
     */
    public JsonNode processAudio(byte[] audioBytes, String userId) throws IOException, InterruptedException {
        logger.info("Processing audio file ({} bytes) for user: {}", audioBytes.length, userId != null ? userId : "default");

        // Create temporary file
        Path tempFile = Files.createTempFile("willspeak_", ".wav");
        Files.write(tempFile, audioBytes);

        try {
            // Build multipart request
            String boundary = "WillSpeakBoundary" + System.currentTimeMillis();
            ByteArrayOutputStream requestBody = new ByteArrayOutputStream();

            // Add user_id part if provided
            if (userId != null && !userId.isEmpty()) {
                String userIdPartHeader =
                        "--" + boundary + "\r\n" +
                                "Content-Disposition: form-data; name=\"user_id\"\r\n\r\n";
                requestBody.write(userIdPartHeader.getBytes());
                requestBody.write(userId.getBytes());
                requestBody.write("\r\n".getBytes());
            }

            // Add file part
            String filePartHeader =
                    "--" + boundary + "\r\n" +
                            "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n" +
                            "Content-Type: audio/wav\r\n\r\n";
            requestBody.write(filePartHeader.getBytes());
            requestBody.write(audioBytes);
            requestBody.write(("\r\n--" + boundary + "--\r\n").getBytes());

            // Select endpoint based on whether user ID is provided
            String endpoint = userId != null ? "/process-audio-for-user" : "/process-audio";

            // Create HTTP request
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(serverUrl + endpoint))
                    .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                    .POST(HttpRequest.BodyPublishers.ofByteArray(requestBody.toByteArray()))
                    .build();

            // Send request and get response
            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200) {
                JsonNode responseJson = objectMapper.readTree(response.body());
                logger.info("Audio processed successfully, transcription: {}",
                        responseJson.path("transcription").asText());
                return responseJson;
            } else {
                logger.error("Error processing audio: HTTP {}", response.statusCode());
                throw new IOException("Server returned status code: " + response.statusCode());
            }
        } finally {
            // Clean up temporary file
            Files.deleteIfExists(tempFile);
        }
    }

    /**
     * Initialize a WebSocket connection for real-time audio processing.
     *
     * @param dataConsumer consumer for processed audio data
     * @return true if connection established, false otherwise
     */
    public boolean startRealTimeProcessing(Consumer<byte[]> dataConsumer) {
        return startRealTimeProcessing(dataConsumer, null);
    }

    /**
     * Initialize a WebSocket connection for real-time audio processing with a specific user profile.
     *
     * @param dataConsumer consumer for processed audio data
     * @param userId the user ID to process with (may be null)
     * @return true if connection established, false otherwise
     */
    public boolean startRealTimeProcessing(Consumer<byte[]> dataConsumer, String userId) {
        logger.info("Starting real-time processing for user: {}", userId != null ? userId : "default");

        if (webSocket != null) {
            logger.warn("WebSocket connection already exists");
            return false;
        }

        this.audioDataConsumer = dataConsumer;

        try {
            // Build WebSocket URI with user ID if provided
            URI uri = URI.create(serverUrl + "/ws" + (userId != null ? "?user_id=" + userId : ""));

            CompletableFuture<WebSocket> webSocketFuture = httpClient.newWebSocketBuilder()
                    .buildAsync(uri, new WebSocketListener());

            webSocket = webSocketFuture.get();
            logger.info("WebSocket connection established");
            return true;

        } catch (InterruptedException e) {
            logger.error("WebSocket connection interrupted", e);
            Thread.currentThread().interrupt();
            return false;
        } catch (ExecutionException e) {
            logger.error("Failed to establish WebSocket connection", e);
            return false;
        }
    }

    /**
     * Send audio data for real-time processing through the WebSocket.
     *
     * @param audioChunk the audio data to process
     * @return true if sent successfully, false otherwise
     */
    public boolean processAudioChunk(byte[] audioChunk) {
        if (webSocket == null) {
            logger.warn("WebSocket not connected");
            return false;
        }

        try {
            webSocket.sendBinary(ByteBuffer.wrap(audioChunk), true);
            return true;
        } catch (Exception e) {
            logger.error("Error sending audio chunk", e);
            return false;
        }
    }

    /**
     * Stop the real-time processing connection.
     */
    public void stopRealTimeProcessing() {
        logger.info("Stopping real-time processing");

        if (webSocket != null) {
            webSocket.sendClose(WebSocket.NORMAL_CLOSURE, "Client closed connection");
            webSocket = null;
            audioDataConsumer = null;
        }
    }

    /**
     * Check server connection.
     *
     * @return true if server is available, false otherwise
     */
    public boolean checkServerConnection() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(serverUrl + "/"))
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            boolean isConnected = response.statusCode() == 200;
            logger.info("Server connection check: {}", isConnected ? "Available" : "Unavailable");
            return isConnected;
        } catch (Exception e) {
            logger.error("Error checking server connection", e);
            return false;
        }
    }

    /**
     * Create a new user profile on the server.
     *
     * @param name The name for the new profile
     * @param description Optional description
     * @return CompletableFuture containing the response JSON
     */
    public CompletableFuture<JsonNode> createUserProfile(String name, String description) {
        CompletableFuture<JsonNode> future = new CompletableFuture<>();

        // Build form data
        Map<String, String> formData = new HashMap<>();
        formData.put("name", name);
        if (description != null) {
            formData.put("description", description);
        }

        // Create and send request asynchronously
        createFormDataRequest("/user/create", formData)
                .thenCompose(request -> httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString()))
                .thenApply(response -> {
                    if (response.statusCode() != 200) {
                        throw new RuntimeException("Server returned status code: " + response.statusCode());
                    }
                    try {
                        return objectMapper.readTree(response.body());
                    } catch (Exception e) {
                        throw new RuntimeException("Error parsing response JSON", e);
                    }
                })
                .whenComplete((result, error) -> {
                    if (error != null) {
                        future.completeExceptionally(error);
                    } else {
                        future.complete(result);
                    }
                });

        return future;
    }

    /**
     * Get all user profiles from the server.
     *
     * @return CompletableFuture containing the response JSON
     */
    public CompletableFuture<JsonNode> getUserProfiles() {
        CompletableFuture<JsonNode> future = new CompletableFuture<>();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/user/list"))
                .GET()
                .build();

        httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() != 200) {
                        throw new RuntimeException("Server returned status code: " + response.statusCode());
                    }
                    try {
                        return objectMapper.readTree(response.body());
                    } catch (Exception e) {
                        throw new RuntimeException("Error parsing response JSON", e);
                    }
                })
                .whenComplete((result, error) -> {
                    if (error != null) {
                        future.completeExceptionally(error);
                    } else {
                        future.complete(result);
                    }
                });

        return future;
    }

    /**
     * Start a new training session.
     *
     * @param userId The user ID to train for
     * @return CompletableFuture containing the response JSON with session ID
     */
    public CompletableFuture<JsonNode> startTrainingSession(String userId) {
        CompletableFuture<JsonNode> future = new CompletableFuture<>();

        // Build form data
        Map<String, String> formData = new HashMap<>();
        formData.put("user_id", userId);

        // Create and send request asynchronously
        createFormDataRequest("/training/start", formData)
                .thenCompose(request -> httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString()))
                .thenApply(response -> {
                    if (response.statusCode() != 200) {
                        throw new RuntimeException("Server returned status code: " + response.statusCode());
                    }
                    try {
                        return objectMapper.readTree(response.body());
                    } catch (Exception e) {
                        throw new RuntimeException("Error parsing response JSON", e);
                    }
                })
                .whenComplete((result, error) -> {
                    if (error != null) {
                        future.completeExceptionally(error);
                    } else {
                        future.complete(result);
                    }
                });

        return future;
    }

    /**
     * Upload a training sample to a training session.
     *
     * @param sessionId The training session ID
     * @param prompt The text prompt that was spoken
     * @param audioFile The audio file containing the speech sample
     * @return CompletableFuture containing the response JSON
     */
    public CompletableFuture<JsonNode> uploadTrainingAudio(String sessionId, String prompt, File audioFile) {
        CompletableFuture<JsonNode> future = new CompletableFuture<>();

        try {
            // Build multipart request
            String boundary = "WillSpeakBoundary" + System.currentTimeMillis();
            ByteArrayOutputStream requestBody = new ByteArrayOutputStream();

            // Add prompt part
            String promptPartHeader =
                    "--" + boundary + "\r\n" +
                            "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n";
            requestBody.write(promptPartHeader.getBytes());
            requestBody.write(prompt.getBytes());
            requestBody.write("\r\n".getBytes());

            // Add file part
            String filePartHeader =
                    "--" + boundary + "\r\n" +
                            "Content-Disposition: form-data; name=\"audio_file\"; filename=\"" +
                            audioFile.getName() + "\"\r\n" +
                            "Content-Type: audio/wav\r\n\r\n";
            requestBody.write(filePartHeader.getBytes());
            requestBody.write(Files.readAllBytes(audioFile.toPath()));
            requestBody.write(("\r\n--" + boundary + "--\r\n").getBytes());

            // Create HTTP request
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(serverUrl + "/training/" + sessionId + "/upload"))
                    .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                    .POST(HttpRequest.BodyPublishers.ofByteArray(requestBody.toByteArray()))
                    .build();

            // Send request asynchronously
            httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                    .thenApply(response -> {
                        if (response.statusCode() != 200) {
                            throw new RuntimeException("Server returned status code: " + response.statusCode());
                        }
                        try {
                            return objectMapper.readTree(response.body());
                        } catch (Exception e) {
                            throw new RuntimeException("Error parsing response JSON", e);
                        }
                    })
                    .whenComplete((result, error) -> {
                        if (error != null) {
                            future.completeExceptionally(error);
                        } else {
                            future.complete(result);
                        }
                    });
        } catch (Exception e) {
            future.completeExceptionally(e);
        }

        return future;
    }

    /**
     * Complete a training session and initiate model training.
     *
     * @param sessionId The training session ID
     * @return CompletableFuture containing the response JSON
     */
    public CompletableFuture<JsonNode> completeTrainingSession(String sessionId) {
        CompletableFuture<JsonNode> future = new CompletableFuture<>();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/training/" + sessionId + "/complete"))
                .POST(HttpRequest.BodyPublishers.noBody())
                .build();

        httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() != 200) {
                        throw new RuntimeException("Server returned status code: " + response.statusCode());
                    }
                    try {
                        return objectMapper.readTree(response.body());
                    } catch (Exception e) {
                        throw new RuntimeException("Error parsing response JSON", e);
                    }
                })
                .whenComplete((result, error) -> {
                    if (error != null) {
                        future.completeExceptionally(error);
                    } else {
                        future.complete(result);
                    }
                });

        return future;
    }

    /**
     * Helper method to create form data HTTP request.
     *
     * @param endpoint The API endpoint
     * @param formData Map of form parameters
     * @return CompletableFuture containing the HTTP request
     */
    private CompletableFuture<HttpRequest> createFormDataRequest(String endpoint, Map<String, String> formData) {
        try {
            // Build form data string
            StringBuilder formDataString = new StringBuilder();
            for (Map.Entry<String, String> entry : formData.entrySet()) {
                if (formDataString.length() > 0) {
                    formDataString.append("&");
                }
                formDataString.append(
                        URLEncoder.encode(entry.getKey(), StandardCharsets.UTF_8));
                formDataString.append("=");
                formDataString.append(
                        URLEncoder.encode(entry.getValue(), StandardCharsets.UTF_8));
            }

            // Create HTTP request
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(serverUrl + endpoint))
                    .header("Content-Type", "application/x-www-form-urlencoded")
                    .POST(HttpRequest.BodyPublishers.ofString(formDataString.toString()))
                    .build();

            return CompletableFuture.completedFuture(request);
        } catch (Exception e) {
            CompletableFuture<HttpRequest> future = new CompletableFuture<>();
            future.completeExceptionally(e);
            return future;
        }
    }

    /**
     * Get the server URL.
     *
     * @return The server URL
     */
    public String getServerUrl() {
        return serverUrl;
    }

    /**
     * WebSocket listener for handling real-time audio processing.
     */
    private class WebSocketListener implements WebSocket.Listener {
        private final ByteArrayOutputStream messageBuffer = new ByteArrayOutputStream();

        @Override
        public void onOpen(WebSocket webSocket) {
            logger.info("WebSocket connection opened");
            WebSocket.Listener.super.onOpen(webSocket);
        }

        @Override
        public CompletionStage<?> onBinary(WebSocket webSocket, ByteBuffer data, boolean last) {
            try {
                byte[] bytes = new byte[data.remaining()];
                data.get(bytes);

                if (audioDataConsumer != null) {
                    audioDataConsumer.accept(bytes);
                }
            } catch (Exception e) {
                logger.error("Error processing binary WebSocket message", e);
            }

            return WebSocket.Listener.super.onBinary(webSocket, data, last);
        }

        @Override
        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
            logger.debug("Received text message: {}", data);
            return WebSocket.Listener.super.onText(webSocket, data, last);
        }

        @Override
        public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
            logger.info("WebSocket closed: {} - {}", statusCode, reason);
            return WebSocket.Listener.super.onClose(webSocket, statusCode, reason);
        }

        @Override
        public void onError(WebSocket webSocket, Throwable error) {
            logger.error("WebSocket error", error);
            WebSocket.Listener.super.onError(webSocket, error);
        }
    }
}