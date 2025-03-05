package com.rednetty.willspeak.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Represents a user profile with speech characteristics and training history.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class UserProfile implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String id;
    private String name;
    private String description;
    private LocalDateTime created;
    private LocalDateTime lastModified;
    private List<TrainingSession> trainingSessions;
    private boolean modelTrained;
    private int totalTrainingAudioSeconds;

    /**
     * Create a new user profile with a random UUID.
     *
     * @param name The name of the user
     */
    public UserProfile(String name) {
        this.id = UUID.randomUUID().toString();
        this.name = name;
        this.description = "";
        this.created = LocalDateTime.now();
        this.lastModified = this.created;
        this.trainingSessions = new ArrayList<>();
        this.modelTrained = false;
        this.totalTrainingAudioSeconds = 0;
    }

    /**
     * Create a new user profile with a specific ID.
     * This is used when creating a local profile for a server-side profile.
     *
     * @param id The ID to use for this profile
     * @param name The name of the user
     */
    public UserProfile(String id, String name) {
        this.id = id;
        this.name = name;
        this.description = "";
        this.created = LocalDateTime.now();
        this.lastModified = this.created;
        this.trainingSessions = new ArrayList<>();
        this.modelTrained = false;
        this.totalTrainingAudioSeconds = 0;
    }

    /**
     * Constructor for Jackson deserialization.
     */
    @JsonCreator
    public UserProfile(
            @JsonProperty("id") String id,
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("created") LocalDateTime created,
            @JsonProperty("lastModified") LocalDateTime lastModified,
            @JsonProperty("trainingSessions") List<TrainingSession> trainingSessions,
            @JsonProperty("modelTrained") boolean modelTrained,
            @JsonProperty("totalTrainingAudioSeconds") int totalTrainingAudioSeconds) {
        this.id = id;
        this.name = name;
        this.description = description != null ? description : "";
        this.created = created != null ? created : LocalDateTime.now();
        this.lastModified = lastModified != null ? lastModified : this.created;
        this.trainingSessions = trainingSessions != null ? trainingSessions : new ArrayList<>();
        this.modelTrained = modelTrained;
        this.totalTrainingAudioSeconds = totalTrainingAudioSeconds;
    }

    // Rest of the class remains the same

    /**
     * Add a new training session to this profile.
     *
     * @param session The training session to add
     */
    public void addTrainingSession(TrainingSession session) {
        trainingSessions.add(session);
        totalTrainingAudioSeconds += session.getTotalAudioSeconds();
        lastModified = LocalDateTime.now();
    }

    /**
     * Get the most recent training session, if any.
     *
     * @return The most recent training session, or null if none exists
     */
    public TrainingSession getLatestTrainingSession() {
        if (trainingSessions.isEmpty()) {
            return null;
        }
        return trainingSessions.get(trainingSessions.size() - 1);
    }

    // Getters and setters

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
        this.lastModified = LocalDateTime.now();
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
        this.lastModified = LocalDateTime.now();
    }

    public LocalDateTime getCreated() {
        return created;
    }

    public LocalDateTime getLastModified() {
        return lastModified;
    }

    public List<TrainingSession> getTrainingSessions() {
        return new ArrayList<>(trainingSessions);
    }

    public boolean isModelTrained() {
        return modelTrained;
    }

    public void setModelTrained(boolean modelTrained) {
        this.modelTrained = modelTrained;
        this.lastModified = LocalDateTime.now();
    }

    public int getTotalTrainingAudioSeconds() {
        return totalTrainingAudioSeconds;
    }

    @Override
    public String toString() {
        return name;
    }
}