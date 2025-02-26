import json
import os
from dotenv import load_dotenv

def format_for_render():
    # Read the original JSON file
    with open('doculink-4db99-firebase-adminsdk-fbsvc-6dcd0f868f.json', 'r') as f:
        creds = json.load(f)
    
    # Convert to single-line JSON string without spaces
    compact_json = json.dumps(creds, separators=(',', ':'))
    
    print("Copy this value for GOOGLE_APPLICATION_CREDENTIALS:")
    print(compact_json)

if __name__ == "__main__":
    format_for_render() 