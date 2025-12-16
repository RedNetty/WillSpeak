package com.rednetty.willspeak.service;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule; // Import the JavaTimeModule
import com.rednetty.willspeak.model.UserProfile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Service for managing user profiles.
 */
public class ProfileManager {
    private static final Logger logger = LoggerFactory.getLogger(ProfileManager.class);

    private final Path profilesDirectory;
    private final ObjectMapper objectMapper;
    private List<UserProfile> profiles;

    /**
     * Create a new profile manager with the default profiles directory.
     */
    public ProfileManager() {
        this(Paths.get(System.getProperty("user.home"), ".willspeak", "profiles"));
    }

    /**
     * Create a new profile manager with a custom profiles directory.
     *
     * @param profilesDirectory The directory to store profiles
     */
    public ProfileManager(Path profilesDirectory) {
        this.profilesDirectory = profilesDirectory;
        this.objectMapper = new ObjectMapper();
        // Configure ObjectMapper to be more lenient with unknown properties
        this.objectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        // Register the JavaTimeModule to support LocalDateTime
        this.objectMapper.registerModule(new JavaTimeModule());
        this.profiles = new ArrayList<>();

        // Ensure profiles directory exists
        try {
            Files.createDirectories(profilesDirectory);
            logger.info("Profiles directory: {}", profilesDirectory);
        } catch (IOException e) {
            logger.error("Failed to create profiles directory", e);
        }

        loadProfiles();
    }

    /**
     * Load all profiles from the profiles directory.
     */
    private void loadProfiles() {
        profiles.clear();

        try {
            List<Path> profileFiles = Files.list(profilesDirectory)
                    .filter(p -> p.toString().endsWith(".json"))
                    .collect(Collectors.toList());

            for (Path profileFile : profileFiles) {
                try {
                    // Check if file is readable and not empty
                    if (Files.size(profileFile) > 0) {
                        UserProfile profile = objectMapper.readValue(profileFile.toFile(), UserProfile.class);
                        if (profile != null && profile.getId() != null && profile.getName() != null) {
                            profiles.add(profile);
                            logger.debug("Loaded profile: {}", profile.getName());
                        } else {
                            logger.warn("Skipped loading invalid profile from {}", profileFile);
                        }
                    } else {
                        logger.warn("Skipped empty profile file: {}", profileFile);
                    }
                } catch (IOException e) {
                    logger.error("Failed to load profile from {}", profileFile, e);
                    // Consider deleting or renaming corrupted files
                    try {
                        Path backupPath = profileFile.resolveSibling(profileFile.getFileName() + ".corrupted");
                        Files.move(profileFile, backupPath);
                        logger.info("Renamed corrupted profile file to {}", backupPath);
                    } catch (IOException moveError) {
                        logger.error("Failed to rename corrupted profile file", moveError);
                    }
                }
            }

            logger.info("Loaded {} profiles", profiles.size());
        } catch (IOException e) {
            logger.error("Failed to list profile files", e);
        }
    }

    /**
     * Save a profile to disk.
     *
     * @param profile The profile to save
     * @return true if saved successfully, false otherwise
     */
    public boolean saveProfile(UserProfile profile) {
        Path profilePath = profilesDirectory.resolve(profile.getId() + ".json");

        try {
            objectMapper.writeValue(profilePath.toFile(), profile);
            logger.info("Saved profile: {}", profile.getName());

            // Update in-memory list
            Optional<UserProfile> existing = profiles.stream()
                    .filter(p -> p.getId().equals(profile.getId()))
                    .findFirst();

            if (existing.isPresent()) {
                profiles.remove(existing.get());
            }
            profiles.add(profile);

            return true;
        } catch (IOException e) {
            logger.error("Failed to save profile: {}", profile.getName(), e);
            return false;
        }
    }

    /**
     * Create a new profile.
     *
     * @param name The name for the new profile
     * @return The created profile
     */
    public UserProfile createProfile(String name) {
        UserProfile profile = new UserProfile(name);
        saveProfile(profile);
        return profile;
    }

    /**
     * Get a profile by ID.
     *
     * @param id The profile ID
     * @return The profile, or null if not found
     */
    public UserProfile getProfile(String id) {
        return profiles.stream()
                .filter(p -> p.getId().equals(id))
                .findFirst()
                .orElse(null);
    }

    /**
     * Delete a profile by ID.
     *
     * @param id The profile ID
     * @return true if deleted successfully, false otherwise
     */
    public boolean deleteProfile(String id) {
        UserProfile profile = getProfile(id);
        if (profile == null) {
            return false;
        }

        Path profilePath = profilesDirectory.resolve(id + ".json");
        try {
            Files.deleteIfExists(profilePath);
            profiles.remove(profile);
            logger.info("Deleted profile: {}", profile.getName());
            return true;
        } catch (IOException e) {
            logger.error("Failed to delete profile: {}", profile.getName(), e);
            return false;
        }
    }

    /**
     * Get all loaded profiles.
     *
     * @return List of profiles
     */
    public List<UserProfile> getProfiles() {
        return new ArrayList<>(profiles);
    }

    /**
     * Refresh profiles from disk.
     */
    public void refreshProfiles() {
        loadProfiles();
    }

    /**
     * Clean up corrupted profile files.
     */
    public void cleanupCorruptedProfiles() {
        try {
            List<Path> profileFiles = Files.list(profilesDirectory)
                    .filter(p -> p.toString().endsWith(".json"))
                    .collect(Collectors.toList());

            for (Path profileFile : profileFiles) {
                try {
                    // Try to read the file to see if it's valid
                    objectMapper.readValue(profileFile.toFile(), UserProfile.class);
                } catch (IOException e) {
                    logger.warn("Found corrupted profile file: {}", profileFile);
                    try {
                        Path backupPath = profileFile.resolveSibling(profileFile.getFileName() + ".corrupted");
                        Files.move(profileFile, backupPath);
                        logger.info("Renamed corrupted profile file to {}", backupPath);
                    } catch (IOException moveError) {
                        logger.error("Failed to rename corrupted profile file", moveError);
                    }
                }
            }
        } catch (IOException e) {
            logger.error("Failed to list profile files for cleanup", e);
        }
    }
}