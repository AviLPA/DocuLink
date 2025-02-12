import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv
from pathlib import Path

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Try to load .env file for local development, but don't fail if it doesn't exist
try:
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        # Print env contents only in development
        print("\nEnvironment variables:")
        with open(env_path) as f:
            env_contents = f.read()
            print(f"Contents of .env file:\n{env_contents}")
except Exception as e:
    print(f"Note: .env file not found or couldn't be loaded. This is normal in production.")

# Get config from environment variables
config = {
    "type": os.getenv("FIREBASE_TYPE", "service_account"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL")
}

# Print relevant environment variables (excluding sensitive data)
print("\nRelevant environment variables after loading:")
print(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")
print(f"PORT: {os.getenv('PORT')}")

# Initialize Firebase Admin
try:
    # Initialize directly with config dictionary instead of file
    cred = credentials.Certificate(config)
    firebase_admin.initialize_app(cred)
    print(f"Successfully initialized Firebase with credentials from environment variables")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    raise

# Initialize Firestore
db = firestore.client() 