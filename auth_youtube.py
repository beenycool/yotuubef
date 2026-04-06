import os
import logging
import tempfile
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv

from src.config.settings import get_config


logger = logging.getLogger(__name__)


def _write_token_file_secure(token_file: str, token_payload: str) -> None:
    """Write OAuth token atomically with owner-only permissions."""
    token_dir = os.path.dirname(token_file) or "."
    os.makedirs(token_dir, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=token_dir, delete=False
        ) as temp_file:
            temp_file.write(token_payload)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = temp_file.name

        os.chmod(temp_path, 0o600)
        os.replace(temp_path, token_file)
        os.chmod(token_file, 0o600)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def authenticate_youtube() -> Credentials | None:
    # Load environment variables
    load_dotenv()

    cfg = get_config()
    secrets_path = cfg.paths.google_client_secrets_file
    token_path = cfg.paths.youtube_token_file
    client_secrets_file = str(secrets_path) if secrets_path else None
    token_file = str(token_path) if token_path else "youtube_token.json"

    # Scopes required for YouTube upload and analytics
    scopes = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.force-ssl",
        "https://www.googleapis.com/auth/youtubepartner",
    ]

    creds = None

    # The file youtube_token.json stores the user's access and refresh tokens
    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, scopes)
            logger.info("Loaded existing credentials from %s", token_file)
        except Exception as e:
            logger.warning("Failed to parse token file %s: %s", token_file, e)
            creds = None

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Refreshing expired credentials")
                creds.refresh(Request())
            except Exception as e:
                logger.warning("Credential refresh failed, re-running auth flow: %s", e)
                creds = None

        if not creds or not creds.valid:
            logger.info(
                "No valid credentials found. Starting new auth flow using %s",
                client_secrets_file,
            )
            if not client_secrets_file or not os.path.exists(client_secrets_file):
                logger.error("Client secrets file '%s' not found.", client_secrets_file)
                logger.error(
                    "Set api.youtube_client_secrets_file in config.yaml, or "
                    "GOOGLE_CLIENT_SECRETS_FILE in the environment."
                )
                return None

            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes
            )
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        _write_token_file_secure(token_file, creds.to_json())
        logger.info("Credentials saved to %s", token_file)

    logger.info("Authentication successful")
    logger.info("You can now use the YouTube features in your application")
    return creds


if __name__ == "__main__":
    authenticate_youtube()
