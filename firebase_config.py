import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
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

# Get credentials from environment variable
creds_json = os.getenv("FIREBASE_CREDENTIALS")
if not creds_json:
    raise ValueError("FIREBASE_CREDENTIALS environment variable not found")

# Print relevant environment variables (excluding sensitive data)
print("\nRelevant environment variables after loading:")
print(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")
print(f"PORT: {os.getenv('PORT')}")

# Add debug logging for the private key format
print("\nPrivate key verification:")
pk = config.get("private_key", "")
print(f"Starts with correct header: {pk.startswith('-----BEGIN PRIVATE KEY-----')}")
print(f"Ends with correct footer: {pk.endswith('-----END PRIVATE KEY-----\n')}")
print("Contains newlines:", "\n" in pk)

try:
    # Parse the JSON string
    config = json.loads(creds_json)
    
    # Initialize Firebase Admin
    cred = credentials.Certificate(config)
    firebase_admin.initialize_app(cred)
    print("Successfully initialized Firebase with credentials")
except json.JSONDecodeError as e:
    print(f"Error parsing credentials JSON: {e}")
    raise
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    raise

# Initialize Firestore
db = firestore.client() 