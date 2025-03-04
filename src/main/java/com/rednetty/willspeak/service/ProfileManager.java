package com.rednetty.willspeak.service;

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
                    UserProfile profile = objectMapper.readValue(profileFile.toFile(), UserProfile.class);
                    profiles.add(profile);
                    logger.debug("Loaded profile: {}", profile.getName());
                } catch (IOException e) {
                    logger.error("Failed to load profile from {}", profileFile, e);
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
}