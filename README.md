# DocuLink

## Setup

1. Clone the repository
2. Create a `.env` file based on `.env.example`
3. Add your Blockfrost API key to the `.env` file
4. Create a Firebase service account and download the credentials
5. Rename the credentials file to `doculink-4db99-firebase-adminsdk-fbsvc-5fc4ba8a7c.json`
6. Run `pip install -r requirements.txt`
7. Run `python app.py`

## Environment Variables

- `BLOCKFROST_API_KEY`: Your Blockfrost API key
- `CARDANO_NETWORK`: The Cardano network to use (mainnet, preprod, preview)

## Firebase Setup

1. Create a Firebase project and download the service account credentials
2. Rename your credentials file to `doculink-4db99-firebase-adminsdk-fbsvc-5fc4ba8a7c.json`
3. Place it in the project root directory
4. For deployment, use the environment variable `GOOGLE_APPLICATION_CREDENTIALS` with the JSON contents 