package com.rednetty.willspeak.model;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Represents a training session with audio samples and performance metrics.
 */
public class TrainingSession implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String id;
    private final LocalDateTime timestamp;
    private final List<TrainingSample> samples;
    private int totalAudioSeconds;
    private boolean completed;
    private TrainingMetrics metrics;

    /**
     * Create a new training session.
     */
    public TrainingSession() {
        this.id = UUID.randomUUID().toString();
        this.timestamp = LocalDateTime.now();
        this.samples = new ArrayList<>();
        this.totalAudioSeconds = 0;
        this.completed = false;
        this.metrics = new TrainingMetrics();
    }

    /**
     * Add a training sample to this session.
     *
     * @param sample The training sample to add
     */
    public void addSample(TrainingSample sample) {
        samples.add(sample);
        totalAudioSeconds += sample.getDurationSeconds();
    }

    /**
     * Mark the training session as completed and update metrics.
     *
     * @param metrics The metrics from model training
     */
    public void complete(TrainingMetrics metrics) {
        this.completed = true;
        this.metrics = metrics;
    }

    // Getters

    public String getId() {
        return id;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public List<TrainingSample> getSamples() {
        return new ArrayList<>(samples);
    }

    public int getTotalAudioSeconds() {
        return totalAudioSeconds;
    }

    public boolean isCompleted() {
        return completed;
    }

    public TrainingMetrics getMetrics() {
        return metrics;
    }

    public int getSampleCount() {
        return samples.size();
    }

    /**
     * Inner class for training sample data.
     */
    public static class TrainingSample implements Serializable {
        private static final long serialVersionUID = 1L;

        private final String id;
        private final String promptText;
        private final String audioPath;
        private final int durationSeconds;

        public TrainingSample(String promptText, String audioPath, int durationSeconds) {
            this.id = UUID.randomUUID().toString();
            this.promptText = promptText;
            this.audioPath = audioPath;
            this.durationSeconds = durationSeconds;
        }

        public String getId() {
            return id;
        }

        public String getPromptText() {
            return promptText;
        }

        public String getAudioPath() {
            return audioPath;
        }

        public int getDurationSeconds() {
            return durationSeconds;
        }
    }

    /**
     * Inner class for training performance metrics.
     */
    public static class TrainingMetrics implements Serializable {
        private static final long serialVersionUID = 1L;

        private double loss;
        private double accuracy;
        private int epochsCompleted;
        private String modelFilePath;

        public TrainingMetrics() {
            this.loss = 0.0;
            this.accuracy = 0.0;
            this.epochsCompleted = 0;
            this.modelFilePath = "";
        }

        public double getLoss() {
            return loss;
        }

        public void setLoss(double loss) {
            this.loss = loss;
        }

        public double getAccuracy() {
            return accuracy;
        }

        public void setAccuracy(double accuracy) {
            this.accuracy = accuracy;
        }

        public int getEpochsCompleted() {
            return epochsCompleted;
        }

        public void setEpochsCompleted(int epochsCompleted) {
            this.epochsCompleted = epochsCompleted;
        }

        public String getModelFilePath() {
            return modelFilePath;
        }

        public void setModelFilePath(String modelFilePath) {
            this.modelFilePath = modelFilePath;
        }
    }
}