package com.rednetty.willspeak.model;

/**
 * Model class for training data pairs (impaired and clear speech).
 */
public class TrainingPair {
    private final String id;
    private final String userId;
    private final String prompt;
    private final String notes;
    private final String impairedPath;
    private final String clearPath;
    private final String created;
    private final String templateId;

    /**
     * Create a new training pair.
     *
     * @param id Unique identifier
     * @param userId The user ID this training pair is for
     * @param prompt The text that was spoken
     * @param notes Optional notes about the speech pattern
     * @param impairedPath Path to the impaired speech audio file
     * @param clearPath Path to the clear speech audio file
     * @param created Creation timestamp
     * @param templateId Optional ID of template used (may be null)
     */
    public TrainingPair(String id, String userId, String prompt, String notes,
                        String impairedPath, String clearPath, String created, String templateId) {
        this.id = id;
        this.userId = userId;
        this.prompt = prompt;
        this.notes = notes;
        this.impairedPath = impairedPath;
        this.clearPath = clearPath;
        this.created = created;
        this.templateId = templateId;
    }

    public String getId() {
        return id;
    }

    public String getUserId() {
        return userId;
    }

    public String getPrompt() {
        return prompt;
    }

    public String getNotes() {
        return notes;
    }

    public String getImpairedPath() {
        return impairedPath;
    }

    public String getClearPath() {
        return clearPath;
    }

    public String getCreated() {
        return created;
    }

    public String getTemplateId() {
        return templateId;
    }

    @Override
    public String toString() {
        return prompt + " (" + created + ")";
    }
}