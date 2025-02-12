import json
import os
from dotenv import load_dotenv

def format_for_render():
    # Read the Firebase JSON file
    with open('doculink-4db99-firebase-adminsdk-fbsvc-6dcd0f868f.json', 'r') as file:
        creds = json.load(file)
    
    # Format with no spaces and ensure correct newlines in private key
    formatted_creds = json.dumps(creds, separators=(',', ':'))
    
    print("\n=== COPY THIS ENTIRE LINE TO FIREBASE_CREDENTIALS IN RENDER ===\n")
    print(formatted_creds)
    print("\n=== END OF CREDENTIALS ===")

if __name__ == "__main__":
    format_for_render() 