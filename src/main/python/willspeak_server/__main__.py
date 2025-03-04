"""
WillSpeak Server - Main entry point
"""
import uvicorn
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from willspeak_server.api.server import app
from willspeak_server.utils.logger import setup_logger

def main():
    """
    Main entry point for the WillSpeak server
    """
    # Set up logging
    logger = setup_logger()
    logger.info("Starting WillSpeak Server")

    # Initialize speech enhancement models
    from willspeak_server.ml.model_integration import initialize_models
    logger.info("Initializing speech enhancement models")
    initialize_models()

    # Run the FastAPI application
    host = os.getenv("WILLSPEAK_HOST", "127.0.0.1")
    port = int(os.getenv("WILLSPEAK_PORT", "8000"))

    logger.info(f"Server running at http://{host}:{port}")

    uvicorn.run(
        "willspeak_server.api.server:app",
        host=host,
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main()