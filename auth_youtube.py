"""
Standalone CLI wrapper for YouTube OAuth authentication.

Delegates all credential logic to
``src.integrations.youtube_client.get_youtube_credentials``.
"""

import logging
from src.integrations.youtube_client import get_youtube_credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    creds = get_youtube_credentials()
    if creds:
        logger.info("Authentication successful")
        logger.info("You can now use the YouTube features in your application")
    else:
        logger.error("Authentication failed")
