import uvicorn
from willspeak_server.api.server import app
from willspeak_server.utils.logger import setup_logger
import os

def main():
    logger = setup_logger()
    logger.info("Starting WillSpeak Server")

    from willspeak_server.ml.model_integration import initialize_models
    logger.info("Initializing speech enhancement models")
    initialize_models()

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