"""
Training Data Collection API Endpoints

This module provides endpoints for managing paired speech samples (impaired and clear)
for training personalized speech enhancement models.
"""
import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import json
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
from loguru import logger
from pathlib import Path

# Import required modules
from willspeak_server.ml.model_integration import train_model

# Create router
training_data_router = APIRouter(prefix="/training-data", tags=["training-data"])

# Get the paths from central configuration
from willspeak_server.api.config import data_dir, user_dir, training_data_dir

# Ensure directories exist
for directory in [data_dir, user_dir, training_data_dir]:
    os.makedirs(directory, exist_ok=True)

# Create jobs directory
jobs_dir = training_data_dir / "jobs"
os.makedirs(jobs_dir, exist_ok=True)


@training_data_router.post("/create-pair")
async def create_training_pair(
        impaired_audio: UploadFile = File(...),
        clear_audio: UploadFile = File(...),
        prompt: str = Form(...),
        user_id: str = Form(...),
        notes: Optional[str] = Form(None)
):
    """
    Create a training pair with both impaired and clear speech for the same prompt.

    Args:
        impaired_audio: The user's impaired speech audio
        clear_audio: The clear speech version (could be from another speaker)
        prompt: The text that was spoken
        user_id: The user ID this training pair is for
        notes: Optional notes about the speech pattern

    Returns:
        Training pair metadata
    """
    # Validate the user exists
    if not await verify_user_exists(user_id):
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    # Create unique ID for this training pair
    pair_id = str(uuid.uuid4())

    # Create directory structure
    user_training_dir = training_data_dir / user_id
    user_training_dir.mkdir(exist_ok=True)

    pair_dir = user_training_dir / pair_id
    pair_dir.mkdir(exist_ok=True)

    # Save audio files
    impaired_path = pair_dir / "impaired.wav"
    clear_path = pair_dir / "clear.wav"

    impaired_content = await impaired_audio.read()
    clear_content = await clear_audio.read()

    with open(impaired_path, "wb") as f:
        f.write(impaired_content)

    with open(clear_path, "wb") as f:
        f.write(clear_content)

    # Create metadata
    metadata = {
        "id": pair_id,
        "user_id": user_id,
        "prompt": prompt,
        "notes": notes,
        "impaired_path": str(impaired_path),
        "clear_path": str(clear_path),
        "created": datetime.now().isoformat()
    }

    # Save metadata
    with open(pair_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created training pair {pair_id} for user {user_id}")
    return metadata


@training_data_router.post("/upload-clear-template")
async def upload_clear_template(
        audio: UploadFile = File(...),
        prompt: str = Form(...),
        speaker_name: str = Form(...),
        category: Optional[str] = Form("general")
):
    """
    Upload a clear speech template that can be used as a reference for multiple users.

    Args:
        audio: Clear speech audio file
        prompt: The text that was spoken
        speaker_name: Name of the speaker
        category: Optional category for organization

    Returns:
        Template metadata
    """
    # Create unique ID for this template
    template_id = str(uuid.uuid4())

    # Create directory structure
    templates_dir = training_data_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    category_dir = templates_dir / category
    category_dir.mkdir(exist_ok=True)

    # Save audio file
    audio_path = category_dir / f"{template_id}.wav"
    content = await audio.read()

    with open(audio_path, "wb") as f:
        f.write(content)

    # Create metadata
    metadata = {
        "id": template_id,
        "prompt": prompt,
        "speaker_name": speaker_name,
        "category": category,
        "audio_path": str(audio_path),
        "created": datetime.now().isoformat()
    }

    # Save metadata
    with open(category_dir / f"{template_id}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created clear speech template {template_id} by {speaker_name}")
    return metadata


@training_data_router.get("/templates")
async def list_templates(category: Optional[str] = None):
    """
    List available clear speech templates.

    Args:
        category: Optional category filter

    Returns:
        List of template metadata
    """
    templates_dir = training_data_dir / "templates"
    if not templates_dir.exists():
        return {"templates": []}

    templates = []

    # If category specified, only look in that directory
    if category:
        category_dir = templates_dir / category
        if not category_dir.exists():
            return {"templates": []}

        dirs_to_search = [category_dir]
    else:
        # Search all category directories
        dirs_to_search = [d for d in templates_dir.iterdir() if d.is_dir()]

    # Collect templates from all matching directories
    for directory in dirs_to_search:
        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    metadata = json.load(f)
                templates.append(metadata)
            except Exception as e:
                logger.error(f"Error loading template metadata from {file_path}: {e}")

    return {"templates": templates}


@training_data_router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """
    Get metadata for a specific template.

    Args:
        template_id: Template ID

    Returns:
        Template metadata
    """
    templates_dir = training_data_dir / "templates"
    if not templates_dir.exists():
        raise HTTPException(status_code=404, detail="Templates directory not found")

    # Search for the template in all category directories
    for category_dir in templates_dir.iterdir():
        if category_dir.is_dir():
            metadata_path = category_dir / f"{template_id}.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading template metadata: {e}")
                    raise HTTPException(status_code=500, detail=f"Error loading template: {str(e)}")

    raise HTTPException(status_code=404, detail="Template not found")


@training_data_router.get("/templates/{template_id}/audio")
async def get_template_audio(template_id: str):
    """
    Get audio file for a specific template.

    Args:
        template_id: Template ID

    Returns:
        Template audio file
    """
    templates_dir = training_data_dir / "templates"
    if not templates_dir.exists():
        raise HTTPException(status_code=404, detail="Templates directory not found")

    # Search for the template in all category directories
    for category_dir in templates_dir.iterdir():
        if category_dir.is_dir():
            metadata_path = category_dir / f"{template_id}.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    audio_path = Path(metadata["audio_path"])
                    if not audio_path.exists():
                        raise HTTPException(status_code=404, detail="Template audio file not found")

                    return FileResponse(audio_path)
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error providing template audio: {e}")
                    raise HTTPException(status_code=500, detail=f"Error providing template audio: {str(e)}")

    raise HTTPException(status_code=404, detail="Template not found")


@training_data_router.get("/user-pairs/{user_id}")
async def list_user_training_pairs(user_id: str):
    """
    List training pairs for a specific user.

    Args:
        user_id: User ID to get training pairs for

    Returns:
        List of training pair metadata
    """
    # Validate the user exists
    if not await verify_user_exists(user_id):
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    user_dir = training_data_dir / user_id
    if not user_dir.exists():
        return {"pairs": []}

    pairs = []

    for pair_dir in user_dir.iterdir():
        if pair_dir.is_dir():
            metadata_path = pair_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    pairs.append(metadata)
                except Exception as e:
                    logger.error(f"Error loading pair metadata from {metadata_path}: {e}")

    return {"pairs": pairs}


@training_data_router.post("/use-template")
async def use_template_for_training(
        impaired_audio: UploadFile = File(...),
        template_id: str = Form(...),
        user_id: str = Form(...),
        notes: Optional[str] = Form(None)
):
    """
    Create a training pair using an existing clear speech template and user's impaired speech.

    Args:
        impaired_audio: The user's impaired speech audio
        template_id: ID of the clear speech template to use
        user_id: The user ID this training pair is for
        notes: Optional notes about the speech pattern

    Returns:
        Training pair metadata
    """
    # Validate the user exists
    if not await verify_user_exists(user_id):
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    # Find the template
    templates_dir = training_data_dir / "templates"
    template_metadata = None
    template_audio_path = None

    # Search for the template in all category directories
    for category_dir in templates_dir.iterdir():
        if category_dir.is_dir():
            metadata_path = category_dir / f"{template_id}.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        template_metadata = json.load(f)
                    template_audio_path = Path(template_metadata["audio_path"])
                    if not template_audio_path.exists():
                        raise HTTPException(status_code=404, detail="Template audio file not found")
                    break
                except Exception as e:
                    logger.error(f"Error loading template metadata: {e}")
                    raise HTTPException(status_code=500, detail=f"Error loading template: {str(e)}")

    if not template_metadata:
        raise HTTPException(status_code=404, detail="Template not found")

    # Create a new training pair
    pair_id = str(uuid.uuid4())

    # Create directory structure
    user_dir = training_data_dir / user_id
    user_dir.mkdir(exist_ok=True)

    pair_dir = user_dir / pair_id
    pair_dir.mkdir(exist_ok=True)

    # Save audio files
    impaired_path = pair_dir / "impaired.wav"
    clear_path = pair_dir / "clear.wav"

    # Save impaired audio from upload
    impaired_content = await impaired_audio.read()
    with open(impaired_path, "wb") as f:
        f.write(impaired_content)

    # Copy clear audio from template
    with open(template_audio_path, "rb") as src, open(clear_path, "wb") as dst:
        dst.write(src.read())

    # Create metadata
    metadata = {
        "id": pair_id,
        "user_id": user_id,
        "prompt": template_metadata["prompt"],
        "template_id": template_id,
        "notes": notes,
        "impaired_path": str(impaired_path),
        "clear_path": str(clear_path),
        "created": datetime.now().isoformat()
    }

    # Save metadata
    with open(pair_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created training pair {pair_id} for user {user_id} using template {template_id}")
    return metadata


@training_data_router.post("/train-from-pairs")
async def train_from_pairs(
        user_id: str = Form(...),
        background_tasks: BackgroundTasks = None
):
    """
    Train a model using the user's training pairs.

    Args:
        user_id: User ID to train model for
        background_tasks: Optional background tasks for async processing

    Returns:
        Status message and training job info
    """
    # Validate the user exists
    if not await verify_user_exists(user_id):
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    # Get all training pairs for this user
    user_dir = training_data_dir / user_id
    if not user_dir.exists():
        raise HTTPException(status_code=404, detail="No training data found for user")

    # Collect all pair paths
    impaired_paths = []
    clear_paths = []

    for pair_dir in user_dir.iterdir():
        if pair_dir.is_dir():
            metadata_path = pair_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    if "impaired_path" in metadata and "clear_path" in metadata:
                        impaired_path = Path(metadata["impaired_path"])
                        clear_path = Path(metadata["clear_path"])

                        if impaired_path.exists() and clear_path.exists():
                            impaired_paths.append(str(impaired_path))
                            clear_paths.append(str(clear_path))
                except Exception as e:
                    logger.error(f"Error loading pair metadata: {e}")

    if not impaired_paths or not clear_paths:
        raise HTTPException(status_code=404, detail="No valid training pairs found")

    # Create a training job
    job_id = str(uuid.uuid4())
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    job_info = {
        "id": job_id,
        "user_id": user_id,
        "status": "pending",
        "created": datetime.now().isoformat(),
        "impaired_paths": impaired_paths,
        "clear_paths": clear_paths,
        "pair_count": len(impaired_paths)
    }

    # Save job info
    with open(job_dir / "job_info.json", "w") as f:
        json.dump(job_info, f, indent=2)

    # Start training in background
    if background_tasks:
        background_tasks.add_task(
            _run_training_job,
            job_id=job_id,
            user_id=user_id,
            impaired_paths=impaired_paths,
            clear_paths=clear_paths
        )
    else:
        # For immediate execution (not recommended for production)
        await _run_training_job(job_id, user_id, impaired_paths, clear_paths)

    logger.info(f"Started training job {job_id} for user {user_id} with {len(impaired_paths)} pairs")
    return {
        "status": "success",
        "message": f"Training started with {len(impaired_paths)} speech pairs",
        "job_id": job_id
    }


@training_data_router.get("/training-jobs/{user_id}")
async def list_training_jobs(user_id: str):
    """
    List training jobs for a user.

    Args:
        user_id: User ID

    Returns:
        List of training jobs
    """
    # Validate the user exists
    if not await verify_user_exists(user_id):
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    if not jobs_dir.exists():
        return {"jobs": []}

    jobs = []

    for job_dir in jobs_dir.iterdir():
        if job_dir.is_dir():
            job_info_path = job_dir / "job_info.json"
            if job_info_path.exists():
                try:
                    with open(job_info_path, "r") as f:
                        job_info = json.load(f)

                    if job_info.get("user_id") == user_id:
                        jobs.append(job_info)
                except Exception as e:
                    logger.error(f"Error loading job info from {job_info_path}: {e}")

    return {"jobs": jobs}


@training_data_router.get("/training-jobs/{job_id}/status")
async def get_training_job_status(job_id: str):
    """
    Get status of a training job.

    Args:
        job_id: Training job ID

    Returns:
        Training job status
    """
    job_dir = jobs_dir / job_id
    job_info_path = job_dir / "job_info.json"

    if not job_info_path.exists():
        raise HTTPException(status_code=404, detail="Training job not found")

    try:
        with open(job_info_path, "r") as f:
            job_info = json.load(f)
        return job_info
    except json.JSONDecodeError as e:
        logger.error(f"Error loading job info from {job_info_path}: {e}")

        # Create a backup of the corrupted file
        try:
            backup_path = job_info_path.with_suffix('.json.corrupted')

            # Read the raw file content
            with open(job_info_path, "r") as f:
                file_content = f.read()

            with open(backup_path, "w") as f:
                f.write(file_content)

            logger.warning(f"Created backup of corrupted job file at {backup_path}")

            # Create a basic valid job info
            basic_job_info = {
                "id": job_id,
                "status": "failed",
                "error": "Job info file was corrupted and has been reset",
                "created": datetime.now().isoformat(),
                "completed": datetime.now().isoformat()
            }

            # Save the basic job info
            with open(job_info_path, "w") as f:
                json.dump(basic_job_info, f, indent=2)

            logger.info(f"Reset corrupted job info file for job {job_id}")
            return basic_job_info

        except Exception as backup_error:
            logger.error(f"Failed to create backup of corrupted job file: {backup_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading job info: {str(e)}. Backup attempt failed: {str(backup_error)}"
            )
    except Exception as e:
        logger.error(f"Error accessing job info file {job_info_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing job info: {str(e)}")


async def _run_training_job(job_id: str, user_id: str, impaired_paths: List[str], clear_paths: List[str]):
    """
    Run a training job in the background.

    Args:
        job_id: Training job ID
        user_id: User ID
        impaired_paths: List of paths to impaired audio files
        clear_paths: List of paths to clear audio files
    """
    job_dir = jobs_dir / job_id
    job_info_path = job_dir / "job_info.json"

    try:
        # Update job status
        with open(job_info_path, "r") as f:
            job_info = json.load(f)

        job_info["status"] = "running"
        job_info["started"] = datetime.now().isoformat()

        with open(job_info_path, "w") as f:
            json.dump(job_info, f, indent=2)

        # Run training
        try:
            # Call the model training function from model_integration.py
            metrics = train_model(impaired_paths, clear_paths, user_id)

            # Convert any NumPy types to native Python types for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    serializable_metrics[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    serializable_metrics[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, list):
                    # Handle lists that might contain NumPy types
                    serializable_list = []
                    for item in value:
                        if isinstance(item, (np.integer, np.int64, np.int32)):
                            serializable_list.append(int(item))
                        elif isinstance(item, (np.floating, np.float64, np.float32)):
                            serializable_list.append(float(item))
                        elif isinstance(item, np.ndarray):
                            serializable_list.append(item.tolist())
                        elif isinstance(item, dict):
                            # Handle dictionaries within lists
                            serializable_dict = {}
                            for k, v in item.items():
                                if isinstance(v, (np.integer, np.int64, np.int32)):
                                    serializable_dict[k] = int(v)
                                elif isinstance(v, (np.floating, np.float64, np.float32)):
                                    serializable_dict[k] = float(v)
                                elif isinstance(v, np.ndarray):
                                    serializable_dict[k] = v.tolist()
                                else:
                                    serializable_dict[k] = v
                            serializable_list.append(serializable_dict)
                        else:
                            serializable_list.append(item)
                    serializable_metrics[key] = serializable_list
                else:
                    serializable_metrics[key] = value

            # Update job with success
            job_info["status"] = "completed"
            job_info["completed"] = datetime.now().isoformat()
            job_info["metrics"] = serializable_metrics

            with open(job_info_path, "w") as f:
                json.dump(job_info, f, indent=2)

            # Update user profile
            user_profile_path = user_dir / user_id / "profile.json"

            with open(user_profile_path, "r") as f:
                profile = json.load(f)

            profile["model_trained"] = True
            profile["last_modified"] = datetime.now().isoformat()

            with open(user_profile_path, "w") as f:
                json.dump(profile, f, indent=2)

            logger.info(f"Training job {job_id} completed successfully")
        except Exception as e:
            # Update job with error
            logger.error(f"Training job {job_id} failed: {e}")
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            job_info["completed"] = datetime.now().isoformat()

            with open(job_info_path, "w") as f:
                json.dump(job_info, f, indent=2)

    except Exception as e:
        logger.error(f"Error in training job {job_id}: {e}")


async def verify_user_exists(user_id: str) -> bool:
    """
    Verify if a user exists in the system.

    Args:
        user_id: The user ID to check

    Returns:
        bool: True if the user exists, False otherwise
    """
    user_path = user_dir / user_id
    profile_path = user_path / "profile.json"

    # Log the verification attempt and paths for debugging
    logger.info(f"Verifying user existence: {user_id}")
    logger.debug(f"Checking path: {profile_path}")

    # Check if the user directory and profile file exist
    if not user_path.exists():
        logger.warning(f"User directory not found: {user_path}")
        return False

    if not profile_path.exists():
        logger.warning(f"User profile not found: {profile_path}")
        return False

    # Validate the profile file
    try:
        with open(profile_path, "r") as f:
            profile = json.load(f)

        # Check if the profile has the required fields
        if "id" not in profile or profile["id"] != user_id:
            logger.warning(f"User profile has invalid ID: {profile.get('id', 'missing')} != {user_id}")
            return False

        logger.info(f"User verified: {user_id} ({profile.get('name', 'unnamed')})")
        return True
    except Exception as e:
        logger.error(f"Error verifying user {user_id}: {str(e)}")
        return False