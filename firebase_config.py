import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv
from pathlib import Path

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Get absolute path to .env file
env_path = Path(__file__).parent / '.env'
print(f".env file exists: {env_path.exists()}")

# Load environment variables from specific file
load_dotenv(dotenv_path=env_path, override=True)

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