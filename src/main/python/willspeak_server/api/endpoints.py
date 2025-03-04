"""
Additional endpoints for WillSpeak server

Adds user profile management and model training endpoints
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import shutil
from pathlib import Path
import json
import uuid
import datetime
from loguru import logger

# Import model integration
from willspeak_server.ml.model_integration import enhance_audio, train_model

# Create router
router = APIRouter()

# Base directories
data_dir = Path(__file__).parent.parent.parent / "data"
user_dir = data_dir / "users"
training_dir = data_dir / "training"

# Ensure directories exist
for directory in [data_dir, user_dir, training_dir]:
    directory.mkdir(exist_ok=True)


@router.post("/user/create")
async def create_user(name: str = Form(...), description: Optional[str] = Form(None)):
    """Create a new user profile"""
    user_id = str(uuid.uuid4())

    # Create user directory
    user_path = user_dir / user_id
    user_path.mkdir(exist_ok=True)

    # Create user profile
    profile = {
        "id": user_id,
        "name": name,
        "description": description or "",
        "created": datetime.datetime.now().isoformat(),
        "last_modified": datetime.datetime.now().isoformat(),
        "training_sessions": [],
        "model_trained": False
    }

    # Save profile
    with open(user_path / "profile.json", "w") as f:
        json.dump(profile, f, indent=2)

    logger.info(f"Created user profile: {name} ({user_id})")
    return profile


@router.get("/user/list")
async def list_users():
    """List all user profiles"""
    users = []

    for user_path in user_dir.iterdir():
        if user_path.is_dir():
            profile_path = user_path / "profile.json"
            if profile_path.exists():
                with open(profile_path, "r") as f:
                    profile = json.load(f)
                    users.append(profile)

    return {"users": users}


@router.get("/user/{user_id}")
async def get_user(user_id: str):
    """Get user profile by ID"""
    user_path = user_dir / user_id
    profile_path = user_path / "profile.json"

    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="User not found")

    with open(profile_path, "r") as f:
        profile = json.load(f)

    return profile


@router.post("/user/{user_id}/update")
async def update_user(user_id: str, name: Optional[str] = Form(None), description: Optional[str] = Form(None)):
    """Update user profile"""
    user_path = user_dir / user_id
    profile_path = user_path / "profile.json"

    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="User not found")

    # Load profile
    with open(profile_path, "r") as f:
        profile = json.load(f)

    # Update fields
    if name:
        profile["name"] = name
    if description is not None:
        profile["description"] = description

    profile["last_modified"] = datetime.datetime.now().isoformat()

    # Save profile
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    logger.info(f"Updated user profile: {profile['name']} ({user_id})")
    return profile


@router.post("/training/start")
async def start_training_session(user_id: str = Form(...)):
    """Start a new training session"""
    user_path = user_dir / user_id
    profile_path = user_path / "profile.json"

    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="User not found")

    # Create session ID
    session_id = str(uuid.uuid4())

    # Create session directory
    session_path = training_dir / session_id
    session_path.mkdir(exist_ok=True)

    # Create session info
    session = {
        "id": session_id,
        "user_id": user_id,
        "started": datetime.datetime.now().isoformat(),
        "completed": False,
        "samples": [],
        "metrics": {}
    }

    # Save session info
    with open(session_path / "session.json", "w") as f:
        json.dump(session, f, indent=2)

    # Update user profile
    with open(profile_path, "r") as f:
        profile = json.load(f)

    profile["training_sessions"].append(session_id)
    profile["last_modified"] = datetime.datetime.now().isoformat()

    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    logger.info(f"Started training session {session_id} for user {user_id}")
    return session


@router.post("/training/{session_id}/upload")
async def upload_training_sample(
        session_id: str,
        prompt: str = Form(...),
        audio_file: UploadFile = File(...),
        clear_audio_file: Optional[UploadFile] = None
):
    """Upload a training sample for a session"""
    session_path = training_dir / session_id

    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Training session not found")

    # Load session info
    with open(session_path / "session.json", "r") as f:
        session = json.load(f)

    # Generate sample ID
    sample_id = str(uuid.uuid4())

    # Save unclear audio
    unclear_path = session_path / f"unclear_{sample_id}.wav"
    with open(unclear_path, "wb") as f:
        f.write(await audio_file.read())

    # Save clear audio if provided
    clear_path = None
    if clear_audio_file:
        clear_path = session_path / f"clear_{sample_id}.wav"
        with open(clear_path, "wb") as f:
            f.write(await clear_audio_file.read())

    # Create sample info
    sample = {
        "id": sample_id,
        "prompt": prompt,
        "unclear_path": str(unclear_path),
        "clear_path": str(clear_path) if clear_path else None,
        "uploaded": datetime.datetime.now().isoformat()
    }

    # Update session
    session["samples"].append(sample)

    with open(session_path / "session.json", "w") as f:
        json.dump(session, f, indent=2)

    logger.info(f"Uploaded training sample {sample_id} for session {session_id}")
    return sample


@router.post("/training/{session_id}/complete")
async def complete_training_session(session_id: str, background_tasks: BackgroundTasks):
    """Complete a training session and train the model"""
    session_path = training_dir / session_id

    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Training session not found")

    # Load session info
    with open(session_path / "session.json", "r") as f:
        session = json.load(f)

    # Check if we have samples
    if not session["samples"]:
        raise HTTPException(status_code=400, detail="No training samples in session")

    # Get user profile
    user_id = session["user_id"]
    user_path = user_dir / user_id
    profile_path = user_path / "profile.json"

    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="User not found")

    # Mark session as completed
    session["completed"] = True
    session["completed_time"] = datetime.datetime.now().isoformat()

    with open(session_path / "session.json", "w") as f:
        json.dump(session, f, indent=2)

    # Prepare training data
    unclear_paths = []
    clear_paths = []

    for sample in session["samples"]:
        unclear_paths.append(sample["unclear_path"])
        if sample["clear_path"]:
            clear_paths.append(sample["clear_path"])
        else:
            # If no clear audio, use unclear as target (for now)
            clear_paths.append(sample["unclear_path"])

    # Start training in background
    background_tasks.add_task(
        _train_model_background,
        unclear_paths=unclear_paths,
        clear_paths=clear_paths,
        user_id=user_id,
        session_id=session_id
    )

    logger.info(f"Completed training session {session_id} for user {user_id}, training started in background")
    return {"status": "success", "message": "Training started in background"}


async def _train_model_background(unclear_paths, clear_paths, user_id, session_id):
    """Background task to train model"""
    session_path = training_dir / session_id

    try:
        # Train model
        metrics = train_model(unclear_paths, clear_paths, user_id)

        # Update session with metrics
        with open(session_path / "session.json", "r") as f:
            session = json.load(f)

        session["metrics"] = metrics

        with open(session_path / "session.json", "w") as f:
            json.dump(session, f, indent=2)

        # Update user profile
        user_path = user_dir / user_id
        profile_path = user_path / "profile.json"

        with open(profile_path, "r") as f:
            profile = json.load(f)

        profile["model_trained"] = True
        profile["last_modified"] = datetime.datetime.now().isoformat()

        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)

        logger.info(f"Model training completed for user {user_id}")

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")

        # Update session with error
        with open(session_path / "session.json", "r") as f:
            session = json.load(f)

        session["error"] = str(e)

        with open(session_path / "session.json", "w") as f:
            json.dump(session, f, indent=2)


@router.post("/process-audio-for-user")
async def process_audio_for_user(user_id: str = Form(...), file: UploadFile = File(...)):
    """Process audio with user-specific model"""
    from willspeak_server.api.server import process_audio_generic

    # Check if user exists
    user_path = user_dir / user_id
    profile_path = user_path / "profile.json"

    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="User not found")

    # Process audio with user's model
    return await process_audio_generic(file, user_id)