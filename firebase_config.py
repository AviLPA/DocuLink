import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Try to load .env file for local development
try:
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        print("Loaded .env file")
except Exception as e:
    print(f"Note: .env file not found or couldn't be loaded. This is normal in production.")

try:
    # Check for credentials in environment variable first (for production/Render)
    firebase_creds_json = os.getenv('FIREBASE_CREDENTIALS')
    
    if firebase_creds_json:
        # Parse the JSON string from environment variable
        print("Using Firebase credentials from environment variable")
        creds_dict = json.loads(firebase_creds_json)
        cred = credentials.Certificate(creds_dict)
    else:
        # Fall back to file for local development
        creds_file = 'doculink-4db99-firebase-adminsdk-fbsvc-5fc4ba8a7c.json'
        if os.path.exists(creds_file):
            print(f"Using Firebase credentials from file: {creds_file}")
            cred = credentials.Certificate(creds_file)
        else:
            raise FileNotFoundError(f"Firebase credentials file not found: {creds_file}")
    
    # Initialize Firebase
    firebase_admin.initialize_app(cred)
    print("Successfully initialized Firebase")
    
except Exception as e:
    print(f"Error initializing Firebase: {str(e)}")
    raise

# Get Firestore database
db = firestore.client() 