import json
import os
from dotenv import load_dotenv

def format_for_render():
    # Read the Firebase JSON file as plain text
    with open('doculink-4db99-firebase-adminsdk-fbsvc-6dcd0f868f.json', 'r') as file:
        content = file.read()
        
    # Remove any whitespace or newlines
    content = content.strip()
    
    print("\n=== COPY THIS ENTIRE LINE TO FIREBASE_CREDENTIALS IN RENDER ===\n")
    print(content)
    print("\n=== END OF CREDENTIALS ===")

if __name__ == "__main__":
    format_for_render() 