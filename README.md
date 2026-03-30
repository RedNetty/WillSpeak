# WillSpeak

**Speech enhancement and communication assistance for individuals with speech impediments.**

WillSpeak is an assistive technology application designed to help people with conditions like Cerebral Palsy communicate more clearly. It uses machine learning to learn and adapt to a user's unique speech patterns, then enhances audio output in real time.

## How It Works

1. **Training Phase** — The user records themselves speaking known phrases. WillSpeak captures these as training pairs (target text → spoken audio).
2. **Model Adaptation** — A personalized speech enhancement model is built from the training data, learning the user's specific vocal characteristics.
3. **Enhancement** — When the user speaks, WillSpeak preprocesses the audio, runs it through the personalized model, and outputs clearer, enhanced speech.
4. **Feedback Loop** — Users can rate enhancement quality to continuously improve their model over time.

## Features

- Real-time audio enhancement via WebSocket
- Personalized per-user speech models
- Training mode with session management
- REST API for integration with other assistive tools
- Batch audio processing endpoint
- MFCC feature extraction and noise reduction pipeline
- Support for WAV, MP3, FLAC, OGG, and M4A audio formats

## Architecture

WillSpeak is split into two components:

| Component | Tech | Purpose |
|-----------|------|---------|
| **Java client** | Spring Boot, JavaFX | Desktop UI, audio capture, user profile management |
| **Python server** | FastAPI, TensorFlow, librosa | ML inference, speech enhancement, REST/WebSocket API |

## Requirements

### Python Server
- Python 3.9+
- See `src/main/python/requirements.txt` for dependencies

### Java Client
- Java 17+
- Maven 3.8+

## Setup

### Python Server

```bash
cd src/main/python

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m willspeak_server
```

The API will be available at `http://localhost:8000`.

### Java Client

```bash
# Build the project
mvn clean install

# Run the application
mvn spring-boot:run
```

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/process-audio` | Enhance an uploaded audio file |
| POST | `/process-audio-for-user` | Enhance with a user-specific model |
| WS | `/ws` | Real-time audio stream enhancement |
| POST | `/users` | Create a user profile |
| POST | `/training/session` | Start a training session |
| GET | `/training/templates` | Get available training phrases |

Full API docs available at `http://localhost:8000/docs` when the server is running.

## Project Structure

```
src/main/
├── java/com/rednetty/willspeak/
│   ├── controller/         # JavaFX controllers (main, training, data)
│   ├── model/              # Domain models (UserProfile, TrainingSession, etc.)
│   ├── service/            # Audio capture, speech processing, profile management
│   └── WillSpeakApp.java   # Application entry point
└── python/willspeak_server/
    ├── api/                # FastAPI routes, server config
    ├── ml/                 # Model integration, speech enhancement model
    ├── speech/             # Audio preprocessing, feature extraction
    └── utils/              # Logger and shared utilities
```

## Motivation

This project was built to address a real gap in assistive communication technology. Existing solutions are often generic and don't adapt to the unique ways individuals with motor speech disorders speak. WillSpeak takes a personalized approach — the more a user trains it, the better it gets for *them specifically*.

## License

See [LICENSE.md](LICENSE.md).
