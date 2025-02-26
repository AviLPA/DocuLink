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
    # Initialize Firebase
    cred = credentials.Certificate('doculink-4db99-firebase-adminsdk-fbsvc-5fc4ba8a7c.json')
    firebase_admin.initialize_app(cred)
    print("Successfully initialized Firebase")
    
except Exception as e:
    print(f"Error initializing Firebase: {str(e)}")
    raise

# Get Firestore database
db = firestore.client() 