import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv
from pathlib import Path


def authenticate_youtube():
    # Load environment variables
    load_dotenv()

    # Configuration
    client_secrets_file = os.getenv("GOOGLE_CLIENT_SECRETS_FILE")
    token_file = os.getenv("YOUTUBE_TOKEN_FILE", "youtube_token.json")

    # Scopes required for YouTube upload and analytics
    scopes = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.force-ssl",
        "https://www.googleapis.com/auth/youtubepartner",
    ]

    creds = None

    # The file youtube_token.json stores the user's access and refresh tokens
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)
        print(f"Loaded existing credentials from {token_file}")

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            print(
                f"No valid credentials found. Starting new auth flow using {client_secrets_file}..."
            )
            if not client_secrets_file or not os.path.exists(client_secrets_file):
                print(f"Error: Client secrets file '{client_secrets_file}' not found.")
                print(
                    "Please ensure GOOGLE_CLIENT_SECRETS_FILE is set correctly in .env"
                )
                return

            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes
            )
            # This will open a browser window for authentication
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_file, "w") as token:
            token.write(creds.to_json())
        print(f"Credentials saved to {token_file}")

    print("\nAuthentication successful!")
    print(f"You can now use the YouTube features in your application.")


if __name__ == "__main__":
    authenticate_youtube()
