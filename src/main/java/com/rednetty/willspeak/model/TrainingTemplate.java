package com.rednetty.willspeak.model;

/**
 * Model class for clear speech templates used in training.
 */
public class TrainingTemplate {
    private final String id;
    private final String prompt;
    private final String speakerName;
    private final String category;
    private final String created;

    /**
     * Create a new training template.
     *
     * @param id Unique identifier
     * @param prompt The text prompt that was spoken
     * @param speakerName Name of the speaker who recorded this template
     * @param category Category for organization (e.g., "vowels", "sentences")
     * @param created Creation timestamp
     */
    public TrainingTemplate(String id, String prompt, String speakerName, String category, String created) {
        this.id = id;
        this.prompt = prompt;
        this.speakerName = speakerName;
        this.category = category;
        this.created = created;
    }

    public String getId() {
        return id;
    }

    public String getPrompt() {
        return prompt;
    }

    public String getSpeakerName() {
        return speakerName;
    }

    public String getCategory() {
        return category;
    }

    public String getCreated() {
        return created;
    }

    @Override
    public String toString() {
        return prompt + " (" + speakerName + ")";
    }
}