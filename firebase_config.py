import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv
from pathlib import Path

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Try to load .env file for local development
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)

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

# Print all environment variables for debugging
print("\nEnvironment variables:")
with open(env_path) as f:
    env_contents = f.read()
    print(f"Contents of .env file:\n{env_contents}")

# Print all relevant environment variables
print("\nRelevant environment variables after loading:")
print(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")
print(f"PORT: {os.getenv('PORT')}")
print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
print(f"FIREBASE_API_KEY: {os.getenv('FIREBASE_API_KEY')}")

# Get the credentials path and ensure it's absolute
cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not os.path.isabs(cred_path):
    cred_path = os.path.join(os.getcwd(), cred_path)

print(f"\nFinal credentials path: {cred_path}")
print(f"Does credentials file exist? {os.path.exists(cred_path)}")

# Initialize Firebase Admin
try:
    print(f"\nAttempting to load credentials from: {cred_path}")
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Credentials file not found at: {cred_path}")
        
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    print(f"Successfully initialized Firebase with credentials from {cred_path}")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    print(f"Attempted credentials path: {cred_path}")
    raise

# Initialize Firestore
db = firestore.client() 