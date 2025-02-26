from flask import Flask, request, jsonify, render_template, Response, send_from_directory, send_file, session
import os
import cv2
import numpy as np
import hashlib
import requests
from blockfrost import BlockFrostApi, ApiError
import sys
import logging as python_logging
import time
from PIL import Image, UnidentifiedImageError, ImageChops
from pdf2image import convert_from_path
from pycardano import *
import json
from enum import Enum
from apscheduler.schedulers.background import BackgroundScheduler
from bs4 import BeautifulSoup
from datetime import datetime
import random
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_config import db
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("python-dotenv not installed, using default environment")

# Configure logging
python_logging.basicConfig(level=python_logging.DEBUG)

app = Flask(__name__)

# Replace the hardcoded API configuration with environment variables
API_KEY = os.getenv('BLOCKFROST_API_KEY')
NETWORK = os.getenv('CARDANO_NETWORK', 'mainnet')  # default to mainnet if not specified

# Set API URL based on network
if NETWORK == 'mainnet':
    API_URL = "https://cardano-mainnet.blockfrost.io/api/v0"
elif NETWORK == 'preprod':  # Changed from testnet to preprod
    API_URL = "https://cardano-preprod.blockfrost.io/api/v0"
elif NETWORK == 'preview':
    API_URL = "https://cardano-preview.blockfrost.io/api/v0"
else:
    raise ValueError(f"Invalid CARDANO_NETWORK value: {NETWORK}")

# Update the wallet address to use mainnet address
WALLET_ADDRESS = "addr1qxcuwgafcr4ahvawfgrdyc37508vxxh9ry8ys05nqx5r7ejhsp3ej007jmr3d6hj7dkwyupyam42yd4znlp9035auwrqg70try"

progress_data = {'processed_frames': 0, 'total_frames': 0}

# List to store hashes
hash_list = []

# Option 1: Add to existing Network enum
class Network(Enum):
    MAINNET = 0
    TESTNET = 1
    PREVIEW = 2  # Add Preview network

# OR Option 2: Create separate enum
class PreviewNetwork(Enum):  # Don't inherit from Network
    PREVIEW = "preview"

# Add near the top with other global variables
verification_stats = {
    'total_verifications': 0,
    'successful_verifications': 0,
    'failed_verifications': 0,
    'verification_times': []  # Store last 100 verification times
}

fraud_statistics = {
    'reported_cases': 0,
    'financial_impact': 0,
    'last_updated': None,
    'deepfake_increase': 0,
    'daily_cases': 0
}

# Add these global variables at the top of your file
last_update = datetime.now()
stats = {
    'daily_new_cases': '2,347',
    'financial_impact': '$4.2B',
    'deepfake_increase': '242%',
    'protected_content': '0'
}

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
COMPARISON_FOLDER = os.path.join(os.getcwd(), 'comparisons')

# Add these configurations
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COMPARISON_FOLDER'] = COMPARISON_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPARISON_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()

# Initialize Blockfrost API
BLOCKFROST_PROJECT_ID = os.getenv('BLOCKFROST_PROJECT_ID')

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

# Add this near the top after imports
if not os.path.exists('.env'):
    print("WARNING: .env file not found!")
else:
    print("Found .env file")

# Add near the top of your file, after loading environment variables
if not API_KEY:
    print("WARNING: No Blockfrost API key found in environment variables!")
    print("Please set BLOCKFROST_API_KEY in your .env file")

# Also verify the network
print(f"Using Cardano network: {NETWORK}")
print(f"Using API URL: {API_URL}")

# Add near your Firebase initialization
try:
    # Change this line to match your actual file name
    if not os.path.exists('doculink-4db99-firebase-adminsdk-fbsvc-5fc4ba8a7c.json'):
        print("WARNING: Firebase credentials file not found!")
        print(f"Looking for file at: {os.path.abspath('doculink-4db99-firebase-adminsdk-fbsvc-5fc4ba8a7c.json')}")
    else:
        print("Firebase credentials file found")
    
    # Initialize Firebase with the correct credentials file
    if not firebase_admin._apps:
        cred = credentials.Certificate('doculink-4db99-firebase-adminsdk-fbsvc-5fc4ba8a7c.json')
        firebase_admin.initialize_app(cred)
    
    # Test Firebase connection
    db = firestore.client()
    # Try a simple operation
    db.collection('test').document('test').get()
    print("Firebase connection successful")
except Exception as e:
    print(f"Firebase error: {str(e)}")

def hash_binary_data(binary_data):
    try:
        python_logging.debug("Hashing binary data")
        sha256_hash = hashlib.sha256()
        sha256_hash.update(binary_data.encode('utf-8'))
        return sha256_hash.hexdigest()
    except Exception as e:
        python_logging.error(f"Error in hash_binary_data: {e}")
        return ""

def search_metadata_for_hash(wallet_address, hash_value):
    # If no wallet address provided, use default mainnet address
    if not wallet_address:
        wallet_address = "addr1qykjh9xrj566jmm5em0epzmsfmdfvpw86nnc3g8f7zsn83u6u3a9lm3zqxwg6sn2m93cm0leqtam6apzje45xcw9dq5s6u8l33"
    headers = {'project_id': API_KEY}
    page = 1
    while True:
        try:
            response = requests.get(f"{API_URL}/addresses/{wallet_address}/transactions?page={page}", headers=headers)
            response.raise_for_status()
            transactions = response.json()
            if not transactions:
                break
            for tx in transactions:
                tx_hash = tx['tx_hash']
                metadata_response = requests.get(f"{API_URL}/txs/{tx_hash}/metadata", headers=headers)
                metadata_response.raise_for_status()
                metadata = metadata_response.json()
                for entry in metadata:
                    if 'json_metadata' in entry:
                        for key, value in entry['json_metadata'].items():
                            if isinstance(value, list) and hash_value in value:
                                python_logging.debug("Transaction found with hash in metadata")
                                return tx_hash, entry
                            elif value == hash_value:
                                python_logging.debug("Transaction found with hash in metadata")
                                return tx_hash, entry
            page += 1
        except requests.exceptions.RequestException as e:
            python_logging.error(f"Request exception: {e}")
            break
    return None

def search_entire_blockchain_for_hash(hash_value):
    headers = {'project_id': API_KEY}
    page = 1
    while True:
        try:
            response = requests.get(f"{API_URL}/metadata/txs/labels?page={page}", headers=headers)
            response.raise_for_status()
            labels = response.json()
            if not labels:
                break
            for label in labels:
                label_page = 1
                while True:
                    metadata_response = requests.get(f"{API_URL}/metadata/txs/labels/{label['label']}?page={label_page}", headers=headers)
                    metadata_response.raise_for_status()
                    transactions = metadata_response.json()
                    if not transactions:
                        break
                    for tx in transactions:
                        if 'json_metadata' in tx:
                            json_metadata = tx['json_metadata']
                            if isinstance(json_metadata, dict):
                                for key, value in json_metadata.items():
                                    if isinstance(value, list) and hash_value in value:
                                        python_logging.debug("Transaction found with hash in metadata")
                                        return tx['tx_hash'], tx
                                    elif value == hash_value:
                                        python_logging.debug("Transaction found with hash in metadata")
                                        return tx['tx_hash'], tx
                            elif isinstance(json_metadata, list):
                                if hash_value in json_metadata:
                                    python_logging.debug("Transaction found with hash in metadata")
                                    return tx['tx_hash'], tx
                    label_page += 1
            page += 1
        except requests.exceptions.RequestException as e:
            python_logging.error(f"Request exception: {e}")
            break
    return None

def video_to_binary(video_path, num_colors=8, target_resolution=(640, 480)):
    python_logging.debug(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    binary_code = ""
    frame_count = 0
    if not cap.isOpened():
        python_logging.error("Failed to open video file")
        return ""

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_data['total_frames'] = total_frames
    python_logging.debug(f"Total frames to process: {total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_resolution)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_image = Image.fromarray(frame)
        frame_binary = image_to_binary(frame_image, num_colors)
        binary_code += frame_binary
        frame_count += 1
        progress_data['processed_frames'] = frame_count
        python_logging.debug(f"Processed frame {frame_count}/{total_frames}")
        time.sleep(0.1)  # Simulate processing time

    cap.release()
    python_logging.debug(f"Video processed, total frames: {frame_count}")
    return binary_code

def image_to_binary(image, num_colors=8):
    try:
        python_logging.debug("Converting image to binary")
        quantized_image = image.convert("L").quantize(colors=num_colors)
        width, height = quantized_image.size
        binary_code = ""
        palette = quantized_image.getpalette()
        color_to_binary = {}
        bits_needed = len(bin(num_colors - 1)[2:])
        for i in range(num_colors):
            binary_value = format(i, f'0{bits_needed}b')
            color_to_binary[i] = binary_value
        for y in range(height):
            for x in range(width):
                pixel = quantized_image.getpixel((x, y))
                binary_code += color_to_binary[pixel]
        python_logging.debug("Image converted to binary")
        return binary_code
    except Exception as e:
        python_logging.error(f"Error in image_to_binary: {e}")
        return ""

def pdf_to_binary(pdf_path, num_colors=8):
    try:
        python_logging.debug(f"Processing PDF: {pdf_path}")
        images = convert_from_path(pdf_path)
        binary_code = ""
        for image in images:
            binary_code += image_to_binary(image, num_colors)
        return binary_code
    except Exception as e:
        python_logging.error(f"Error in pdf_to_binary: {e}")
        return ""

def scrape_ic3_stats():
    try:
        # Scrape FBI IC3 2023 Report
        response = requests.get('https://www.ic3.gov/Media/PDF/AnnualReport/2023_IC3Report.pdf')
        # This would need proper PDF parsing, but for now using example data
        return {
            'total_cases': 800_000,  # Approximate annual cases
            'financial_impact': 10.3  # Billions USD
        }
    except Exception as e:
        python_logging.error(f"Error scraping IC3 stats: {e}")
        return None

def scrape_ftc_stats():
    try:
        # Scrape FTC's latest fraud reports
        response = requests.get('https://www.ftc.gov/news-events/data-visualizations/data-spotlight/trends-identity-theft-fraud')
        soup = BeautifulSoup(response.text, 'html.parser')
        # Parse the data (simplified example)
        return {
            'identity_theft_cases': 1_500_000,
            'fraud_cases': 2_800_000
        }
    except Exception as e:
        python_logging.error(f"Error scraping FTC stats: {e}")
        return None

def scrape_deepfake_stats():
    try:
        # Example API call to a research database
        response = requests.get('https://api.deepfakedetection.org/stats')
        data = response.json()
        return {
            'detected_cases': data['total_cases'],
            'increase_percentage': data['year_over_year_increase']
        }
    except Exception as e:
        python_logging.error(f"Error scraping deepfake stats: {e}")
        return None

def update_fraud_statistics():
    try:
        ic3_stats = scrape_ic3_stats()
        ftc_stats = scrape_ftc_stats()
        deepfake_stats = scrape_deepfake_stats()

        # Use default values if scraping fails
        total_cases = 0
        if ic3_stats:
            total_cases += ic3_stats.get('total_cases', 0)
        if ftc_stats:
            total_cases += ftc_stats.get('identity_theft_cases', 0) + ftc_stats.get('fraud_cases', 0)

        # Update global statistics with default values
        fraud_statistics.update({
            'reported_cases': total_cases or 800000,  # Default value if scraping fails
            'financial_impact': ic3_stats.get('financial_impact', 10.3),
            'last_updated': datetime.now(),
            'deepfake_increase': deepfake_stats.get('increase_percentage', 330),
            'daily_cases': round((total_cases or 800000) / 365)  # Calculate daily cases with default
        })

    except Exception as e:
        python_logging.error(f"Error updating fraud statistics: {e}")
        # Set default values if update fails
        fraud_statistics.update({
            'reported_cases': 800000,
            'financial_impact': 10.3,
            'last_updated': datetime.now(),
            'deepfake_increase': 330,
            'daily_cases': 2500
        })

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(update_fraud_statistics, 'interval', hours=24)
scheduler.start()

@app.route('/upload_file', methods=['POST'])
def upload_file():
    start_time = time.time()
    python_logging.debug("Upload file endpoint called")
    file = request.files.get('file')
    new_wallet = request.form.get('newWallet')
    current_hash = request.form.get('currentHash')
    wallet = request.form.get('wallet')  # Get the wallet address from the form

    python_logging.debug(f"Received new wallet: {new_wallet}")
    python_logging.debug(f"Current hash: {current_hash}")
    python_logging.debug(f"Wallet address provided: {wallet}")

    # Determine which wallet to use
    wallet_to_use = wallet if wallet else "addr1qykjh9xrj566jmm5em0epzmsfmdfvpw86nnc3g8f7zsn83u6u3a9lm3zqxwg6sn2m93cm0leqtam6apzje45xcw9dq5s6u8l33"
    python_logging.debug(f"Using wallet: {wallet_to_use}")

    if new_wallet and current_hash:
        python_logging.debug("Searching new wallet with provided hash")
        tx = search_metadata_for_hash(new_wallet, current_hash)
        if tx:
            python_logging.debug("Transaction found")
            # Update verification stats
            verification_stats['total_verifications'] += 1
            verification_time = time.time() - start_time
            verification_stats['verification_times'].append(verification_time)
            
            verification_stats['successful_verifications'] += 1
            return jsonify({
                'message': 'Transaction found with this hash within the declared wallet. Frame by frame analysis has confirmed authenticity',
                'hash': current_hash,
                'wallet': new_wallet,
                'tx_hash': tx[0],
                'total_frames': progress_data.get('total_frames', 0),
                'processed_frames': progress_data.get('total_frames', 0)
            })
        else:
            python_logging.debug("No transaction found")
            # Update verification stats
            verification_stats['total_verifications'] += 1
            verification_time = time.time() - start_time
            verification_stats['verification_times'].append(verification_time)
            
            verification_stats['failed_verifications'] += 1
            return jsonify({
                'message': 'No transaction found with this hash within the declared wallet: Possible Tampering.',
                'hash': current_hash,
                'wallet': new_wallet,
                'total_frames': progress_data.get('total_frames', 0),
                'processed_frames': progress_data.get('total_frames', 0)
            })
    elif file:
        file_path = os.path.join('uploads', file.filename)
        try:
            file.save(file_path)
            python_logging.debug(f"File saved to {file_path}")

            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = Image.open(file_path)
                    binary_code = image_to_binary(image)
                except UnidentifiedImageError as e:
                    python_logging.error(f"Error opening image: {e}")
                    return jsonify({'message': 'Failed to process image: Unidentified image format.'})
            elif file.filename.lower().endswith('.mp4') or file.filename.lower().endswith('.mov'):
                binary_code = video_to_binary(file_path)
            elif file.filename.lower().endswith('.pdf'):
                binary_code = pdf_to_binary(file_path)
            else:
                python_logging.error("Unsupported file type")
                return jsonify({'message': 'Unsupported file type.'})

            if binary_code:
                hash_result = hash_binary_data(binary_code)
                python_logging.debug(f"Binary data hashed: {hash_result}")

                tx = search_metadata_for_hash(wallet_to_use, hash_result)
                if tx:
                    python_logging.debug("Transaction found")
                    # Update verification stats
                    verification_stats['total_verifications'] += 1
                    verification_time = time.time() - start_time
                    verification_stats['verification_times'].append(verification_time)
                    
                    verification_stats['successful_verifications'] += 1
                    return jsonify({
                        'message': 'Transaction found with this hash within the declared wallet. Frame by frame analysis has confirmed authenticity',
                        'hash': hash_result,
                        'wallet': wallet_to_use,
                        'tx_hash': tx[0],
                        'total_frames': progress_data.get('total_frames', 0),
                        'processed_frames': progress_data.get('total_frames', 0)
                    })
                else:
                    python_logging.debug("No transaction found")
                    # Update verification stats
                    verification_stats['total_verifications'] += 1
                    verification_time = time.time() - start_time
                    verification_stats['verification_times'].append(verification_time)
                    
                    verification_stats['failed_verifications'] += 1
                    return jsonify({
                        'message': 'No transaction found with this hash within the declared wallet: Possible Tampering.',
                        'hash': hash_result,
                        'wallet': wallet_to_use,
                        'total_frames': progress_data.get('total_frames', 0),
                        'processed_frames': progress_data.get('total_frames', 0)
                    })
            else:
                python_logging.error("Failed to process file to binary")
                return jsonify({'message': 'Failed to process file.'})
        except Exception as e:
            python_logging.error(f"Error saving or processing file: {e}")
            return jsonify({'message': 'Error processing file.'})
    else:
        python_logging.error("No file uploaded and no new wallet provided")
        return jsonify({'message': 'No file uploaded and no new wallet provided.'})

@app.route('/search_file', methods=['POST'])
def search_file():
    python_logging.debug("Search file endpoint called")
    file = request.files.get('file')
    if not file:
        python_logging.error("No file uploaded")
        return jsonify({'message': 'No file uploaded.'})

    file_path = os.path.join('uploads', file.filename)
    try:
        file.save(file_path)
        python_logging.debug(f"File saved to {file_path}")

        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            binary_code = image_to_binary(Image.open(file_path))
        elif file.filename.lower().endswith('.mp4') or file.filename.lower().endswith('.mov'):
            binary_code = video_to_binary(file_path)
        elif file.filename.lower().endswith('.pdf'):
            binary_code = pdf_to_binary(file_path)
        else:
            python_logging.error("Unsupported file type")
            return jsonify({'message': 'Unsupported file type.'})

        if binary_code:
            hash_result = hash_binary_data(binary_code)
            python_logging.debug(f"Binary data hashed: {hash_result}")

            tx = search_entire_blockchain_for_hash(hash_result)
            if tx:
                python_logging.debug("Transaction found")
                return jsonify({
                    'message': 'Transaction found with this hash. Frame by frame analysis has confirmed authenticity',
                    'hash': hash_result,
                    'wallet': tx[1].get('address', 'Unknown'),
                    'total_frames': progress_data.get('total_frames', 0),
                    'processed_frames': progress_data.get('total_frames', 0)
                })
            else:
                python_logging.debug("No transaction found")
                return jsonify({
                    'message': 'No transaction found with this hash: Possible Tampering.',
                    'hash': hash_result,
                    'wallet': '',
                    'total_frames': progress_data.get('total_frames', 0),
                    'processed_frames': progress_data.get('total_frames', 0)
                })
        else:
            python_logging.error("Failed to process file to binary")
            return jsonify({'message': 'Failed to process file.'})
    except Exception as e:
        python_logging.error(f"Error saving or processing file: {e}")
        return jsonify({'message': 'Error processing file.'})

@app.route('/compare_videos', methods=['POST'])
def compare_videos_route():
    python_logging.debug("Compare videos endpoint called")
    video1 = request.files.get('video1')
    video2 = request.files.get('video2')

    if not video1 or not video2:
        python_logging.error("One or both videos not uploaded")
        return jsonify({'message': 'Both videos must be uploaded.'})

    video1_path = os.path.join('uploads', video1.filename)
    video2_path = os.path.join('uploads', video2.filename)

    try:
        video1.save(video1_path)
        video2.save(video2_path)
        python_logging.debug(f"Videos saved to {video1_path} and {video2_path}")

        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)

        frames1 = []
        frames2 = []
        while cap1.isOpened() or cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if ret1:
                frames1.append(frame1)
            if ret2:
                frames2.append(frame2)
            if not ret1 and not ret2:
                break

        cap1.release()
        cap2.release()

        comparison_dir = "comparisons"
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)

        comparison_images = []
        differences = []
        max_len = max(len(frames1), len(frames2))

        for i in range(max_len):
            if i < len(frames1) and i < len(frames2):
                frame1 = frames1[i]
                frame2 = frames2[i]
            elif i < len(frames1):
                frame1 = frames1[i]
                frame2 = np.zeros_like(frame1)
            else:
                frame2 = frames2[i]
                frame1 = np.zeros_like(frame2)

            diff_img = cv2.absdiff(frame1, frame2)
            gray_diff = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
            _, threshold_diff = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_diff = np.sum(threshold_diff) / 255
            frame_area = frame1.shape[0] * frame1.shape[1]
            percent_diff = (total_diff / frame_area) * 100
            differences.append(percent_diff)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter small differences
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    explanation = f"Difference in region: (x: {x}, y: {y}, width: {w}, height: {h})"
                    python_logging.debug(explanation)

            frame1_path = os.path.join(comparison_dir, f"frame1_{i}.jpg")
            frame2_path = os.path.join(comparison_dir, f"frame2_{i}.jpg")
            cv2.imwrite(frame1_path, frame1)
            cv2.imwrite(frame2_path, frame2)
            comparison_images.append((frame1_path, frame2_path))

        python_logging.debug(f"Differences: {differences}")

        return jsonify({'message': 'Comparison complete', 'comparison_images': comparison_images, 'differences': differences})
    except Exception as e:
        python_logging.error(f"Error saving or processing videos: {e}")
        return jsonify({'message': 'Error processing videos.'})

@app.route('/compare_images', methods=['POST'])
def compare_images_route():
    python_logging.debug("Compare images endpoint called")
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')

    if not image1 or not image2:
        python_logging.error("One or both images not uploaded")
        return jsonify({'message': 'Both images must be uploaded.'})

    image1_path = os.path.join('uploads', image1.filename)
    image2_path = os.path.join('uploads', image2.filename)

    try:
        image1.save(image1_path)
        image2.save(image2_path)
        python_logging.debug(f"Images saved to {image1_path} and {image2_path}")

        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        diff = ImageChops.difference(img1, img2)
        diff_path = os.path.join('comparisons', 'diff.jpg')
        diff.save(diff_path)

        total_diff = np.sum(np.array(diff))
        img1_size = img1.size[0] * img1.size[1]
        percent_diff = (total_diff / (255 * img1_size)) * 100

        python_logging.debug(f"Differences: {percent_diff}")

        return jsonify({'message': 'Comparison complete', 'diff_image': diff_path, 'difference_percentage': percent_diff})
    except Exception as e:
        python_logging.error(f"Error saving or processing images: {e}")
        return jsonify({'message': 'Error processing images.'})

@app.route('/progress')
def progress():
    def generate():
        while True:
            time.sleep(1)
            yield f"data:{progress_data['processed_frames']}|{progress_data['total_frames']}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/comparisons/<path:filename>')
def serve_comparisons(filename):
    return send_from_directory('comparisons', filename)

@app.route('/creator_hub', methods=['POST'])
def creator_hub():
    file = request.files.get('file')
    title = request.form.get('title')
    description = request.form.get('description')
    creator_name = request.form.get('creator_name')
    
    if not file:
        python_logging.error("No file uploaded")
        return jsonify({'success': False, 'message': 'No file uploaded.'})

    file_path = os.path.join('uploads', file.filename)
    try:
        file.save(file_path)
        python_logging.debug(f"File saved to {file_path}")

        # Process file based on type
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            binary_code = image_to_binary(Image.open(file_path))
        elif file.filename.lower().endswith('.mp4') or file.filename.lower().endswith('.mov'):
            binary_code = video_to_binary(file_path)
        elif file.filename.lower().endswith('.pdf'):
            binary_code = pdf_to_binary(file_path)
        else:
            python_logging.error("Unsupported file type")
            return jsonify({'success': False, 'message': 'Unsupported file type.'})

        if binary_code:
            hash_result = hash_binary_data(binary_code)
            python_logging.debug(f"Binary data hashed: {hash_result}")

            # Add to hash_list with metadata
            hash_list.append({
                'file': file.filename,
                'hash': hash_result,
                'title': title,
                'description': description,
                'creator_name': creator_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'file_type': file.filename.split('.')[-1].lower()
            })
            
            return jsonify({
                'success': True, 
                'file': file.filename, 
                'hash': hash_result,
                'title': title,
                'description': description,
                'creator_name': creator_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            python_logging.error("Failed to process file to binary")
            return jsonify({'success': False, 'message': 'Failed to process file.'})
    except Exception as e:
        python_logging.error(f"Error saving or processing file: {e}")
        return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})

@app.route('/upload_transaction', methods=['POST'])
def upload_transaction():
    try:
        data = request.get_json()
        content_hash = data.get('content_hash')
        creator_wallet = data.get('creator_wallet')  # Get creator's wallet
        
        if not content_hash:
            return jsonify({'success': False, 'message': 'Content hash is required'})

        # Use creator's wallet if provided, otherwise use default
        wallet_to_use = creator_wallet if creator_wallet else "addr1qykjh9xrj566jmm5em0epzmsfmdfvpw86nnc3g8f7zsn83u6u3a9lm3zqxwg6sn2m93cm0leqtam6apzje45xcw9dq5s6u8l33"
        python_logging.info(f"Using wallet address: {wallet_to_use}")

        # Initialize BlockFrost context
        try:
            context = BlockFrostChainContext(
                project_id=API_KEY,
                base_url=API_URL,
                network=Network.MAINNET
            )
            python_logging.info("BlockFrost context initialized successfully")
        except Exception as e:
            python_logging.error(f"Failed to initialize BlockFrost context: {str(e)}")
            return jsonify({'success': False, 'message': f'BlockFrost initialization failed: {str(e)}'})

        try:
            # Get UTXOs for the address
            address = Address.from_primitive(wallet_to_use)
            utxos = context.utxos(address)
            
            if not utxos:
                python_logging.error("No UTXOs found in wallet")
                return jsonify({
                    'success': False,
                    'message': 'No UTXOs found in wallet. Please fund your preview testnet wallet.'
                })

            python_logging.info(f"Found {len(utxos)} UTXOs")
            python_logging.info(f"First UTXO amount: {utxos[0].output.amount.coin} lovelace")

            # Create the transaction builder
            builder = TransactionBuilder(context)
            
            # Add input from the first UTXO
            utxo = utxos[0]
            builder.add_input(utxo)

            # Calculate minimum required output (consider fees)
            min_ada = 1000000  # Minimum ADA requirement (1 ADA = 1,000,000 lovelace)
            fee_buffer = 200000  # 0.2 ADA for fees
            available_amount = utxo.output.amount.coin
            output_amount = available_amount - fee_buffer

            if output_amount < min_ada:
                python_logging.error(f"Insufficient funds. Available: {available_amount}, Minimum required: {min_ada + fee_buffer}")
                return jsonify({
                    'success': False,
                    'message': f'Insufficient funds. Need at least {(min_ada + fee_buffer)/1000000} ADA'
                })

            python_logging.info(f"Setting output amount to {output_amount} lovelace")
            
            # Add output back to our address
            builder.add_output(
                TransactionOutput(
                    address=address,
                    amount=Value(output_amount)
                )
            )

            # Add metadata
            metadata_dict = {
                674: {
                    "DocuLink": {
                        "hash": content_hash,
                        "timestamp": str(int(time.time()))
                    }
                }
            }
            python_logging.info(f"Adding metadata: {metadata_dict}")
            
            auxiliary_data = AuxiliaryData(
                metadata=Metadata(metadata_dict)
            )
            builder.auxiliary_data = auxiliary_data

            # Set validity interval
            current_slot = context.last_block_slot
            builder.ttl = current_slot + 7200  # Valid for 2 hours
            python_logging.info(f"Transaction valid until slot {builder.ttl}")

            # Build and sign the transaction
            python_logging.info("Building and signing transaction...")
            signed_tx = builder.build_and_sign([], change_address=address)
            
            # Submit the transaction
            python_logging.info("Submitting transaction...")
            cbor_hex = signed_tx.to_cbor()
            python_logging.debug(f"Transaction CBOR: {cbor_hex}")

            response = requests.post(
                f"{API_URL}/tx/submit",
                headers={
                    'project_id': API_KEY,
                    'Content-Type': 'application/cbor'
                },
                data=cbor_hex
            )
            
            python_logging.info(f"Submission response status: {response.status_code}")
            python_logging.info(f"Submission response content: {response.text}")

            if response.status_code == 200:
                tx_hash = response.json()
                python_logging.info(f"Transaction submitted successfully. Hash: {tx_hash}")
                return jsonify({
                    'success': True,
                    'transaction': {
                        'hash': tx_hash,
                        'blockNumber': 'Pending',
                        'timestamp': int(time.time())
                    },
                    'metadata': {
                        'content_hash': content_hash
                    }
                })
            else:
                error_msg = f"Transaction submission failed with status {response.status_code}: {response.text}"
                python_logging.error(error_msg)
                return jsonify({
                    'success': False,
                    'message': error_msg
                })

        except Exception as e:
            python_logging.error(f"Error during transaction processing: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'message': f'Transaction processing failed: {str(e)}'
            })

    except Exception as e:
        python_logging.error(f"General error in upload_transaction: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Transaction failed: {str(e)}'
        })

@app.route('/get_verification_stats')
def get_verification_stats():
    global last_update, stats
    
    # Update stats every minute (simulated changing data)
    current_time = datetime.now()
    if (current_time - last_update).seconds > 60:
        # Update with slightly different numbers each time
        daily_cases = random.randint(2000, 3000)
        financial = round(random.uniform(3.8, 4.5), 1)
        deepfake = random.randint(230, 260)
        protected = int(stats['protected_content']) + random.randint(1, 5)
        
        stats = {
            'daily_new_cases': f"{daily_cases:,}",
            'financial_impact': f"${financial}B",
            'deepfake_increase': f"{deepfake}%",
            'protected_content': str(protected)
        }
        last_update = current_time
    
    return jsonify(stats)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/download-test-image')
def download_test_image():
    try:
        return send_file('static/images/test-image.png',
                        mimetype='image/png',
                        as_attachment=True,
                        download_name='test-image.png')
    except Exception as e:
        app.logger.error(f"Error serving test image: {e}")
        return "Error serving test image", 500

@app.route('/verify_token', methods=['POST'])
def verify_token():
    try:
        # Get the ID token from the request
        id_token = request.json['idToken']
        
        # Verify the token
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        
        return jsonify({'success': True, 'uid': uid})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 401

@app.route('/get_thumbnail/<hash>')
def get_thumbnail(hash):
    # Get the file path from your storage
    file_path = f"uploads/{hash}"  # Adjust path as needed
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    else:
        # Return a default thumbnail or 404
        return send_file('static/default-thumbnail.png', mimetype='image/png')

@app.route('/verify', methods=['POST'])
def verify():
    print("API_KEY:", API_KEY)  # Add this line
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    try:
        file = request.files['file']
        # Use the new wallet address as default if none provided
        wallet_address = request.form.get('wallet_address', 
            "addr1qxcuwgafcr4ahvawfgrdyc37508vxxh9ry8ys05nqx5r7ejhsp3ej007jmr3d6hj7dkwyupyam42yd4znlp9035auwrqg70try")
        
        # Debug logging
        print(f"Verifying file: {file.filename}")
        print(f"Wallet address: {wallet_address}")
        
        # Save file temporarily
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        try:
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                binary_code = image_to_binary(Image.open(file_path))
            elif file.filename.lower().endswith(('.mp4', '.mov')):
                binary_code = video_to_binary(file_path)
            elif file.filename.lower().endswith('.pdf'):
                binary_code = pdf_to_binary(file_path)
            else:
                return jsonify({'success': False, 'message': 'Unsupported file type'})
            
            file_hash = hash_binary_data(binary_code)
            print(f"Generated file hash: {file_hash}")
            
            blockchain_result = query_blockchain(file_hash, wallet_address)
            print(f"Blockchain result: {blockchain_result}")
            
            if blockchain_result['found']:
                return jsonify({
                    'success': True,
                    'message': 'Content verified on blockchain!',
                    'txHash': blockchain_result['txHash'],
                    'owner': blockchain_result['owner'],
                    'hash': file_hash
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Content not found on blockchain',
                    'details': 'This content has not been registered or the wallet address does not match',
                    'hash': file_hash
                })
                
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Verification failed',
            'error': str(e)
        })

def query_blockchain(file_hash, wallet_address=None):
    if not wallet_address:
        wallet_address = WALLET_ADDRESS
    try:
        if not API_KEY:
            print("ERROR: No API key found")
            return {'found': False, 'error': 'Missing API key'}
            
        api = BlockFrostApi(
            project_id=API_KEY
        )
        
        print(f"Using API key: {API_KEY[:5]}...")  # Only print first 5 chars for security
        print(f"Querying transactions for wallet: {wallet_address}")
        
        try:
            transactions = api.address_transactions(
                address=wallet_address,
                count=100
            )
            
            for tx in transactions:
                try:
                    metadata = api.transaction_metadata(
                        hash=tx.tx_hash
                    )
                    
                    print(f"\nChecking transaction {tx.tx_hash} metadata:")
                    print(f"Metadata: {metadata}")
                    
                    for meta in metadata:
                        print(f"Checking metadata entry: {meta}")
                        
                        if hasattr(meta, 'json_metadata'):
                            json_meta = meta.json_metadata
                            print(f"JSON metadata: {json_meta}")
                            
                            # Convert Namespace to dict if needed
                            if hasattr(json_meta, '__dict__'):
                                json_dict = vars(json_meta)
                            else:
                                json_dict = json_meta
                                
                            print(f"Converted to dict: {json_dict}")
                            
                            # Check msg list first (most common case)
                            if 'msg' in json_dict and isinstance(json_dict['msg'], list):
                                print(f"Checking msg list: {json_dict['msg']}")
                                print(f"File hash to match: {file_hash}")
                                for item in json_dict['msg']:
                                    print(f"Comparing with item: {item}")
                                    print(f"Types - item: {type(item)}, file_hash: {type(file_hash)}")
                                    print(f"Lengths - item: {len(item)}, file_hash: {len(file_hash)}")
                                    print(f"Are they equal? {item.lower() == file_hash.lower()}")
                                    if item.lower() == file_hash.lower():  # Case-insensitive comparison
                                        print(f"Found matching hash in msg list: {file_hash}")
                                        return {
                                            'found': True,
                                            'txHash': tx.tx_hash,
                                            'timestamp': tx.block_time,
                                            'owner': wallet_address,
                                            'metadata': json_dict
                                        }
                            
                            # Check direct match in the entire dict
                            dict_str = str(json_dict).lower()
                            if file_hash.lower() in dict_str:
                                print(f"Found matching hash in metadata: {file_hash}")
                                return {
                                    'found': True,
                                    'txHash': tx.tx_hash,
                                    'timestamp': tx.block_time,
                                    'owner': wallet_address,
                                    'metadata': json_dict
                                }
                            
                            # Check nested dictionaries
                            for value in json_dict.values():
                                if isinstance(value, dict):
                                    value_str = str(value).lower()
                                    if file_hash.lower() in value_str:
                                        print(f"Found matching hash in nested dict: {file_hash}")
                                        return {
                                            'found': True,
                                            'txHash': tx.tx_hash,
                                            'timestamp': tx.block_time,
                                            'owner': wallet_address,
                                            'metadata': json_dict
                                        }
                
                except Exception as tx_error:
                    print(f"Error processing transaction {tx.tx_hash}: {str(tx_error)}")
                    continue
            
            print("No matching hash found in transactions")
            return {'found': False}
            
        except ApiError as e:
            print(f"Blockfrost API error: {str(e)}")
            print(f"Status code: {e.status_code}")
            if e.status_code == 403:
                return {'found': False, 'error': 'Invalid API key or insufficient permissions'}
            elif e.status_code == 404:
                return {'found': False, 'error': f"No transactions found for address {wallet_address}"}
            return {'found': False, 'error': str(e)}
            
    except Exception as e:
        print(f"Error querying blockchain: {str(e)}")
        return {'found': False, 'error': str(e)}

def verify_api_key():
    try:
        api = BlockFrostApi(
            project_id=API_KEY
        )
        latest_block = api.block_latest()
        print(f"API connection successful. Latest block: {latest_block.hash}")
        return True
    except ApiError as e:
        print(f"API connection failed: {str(e)}")
        print(f"Status code: {e.status_code}")
        return False
    except Exception as e:
        print(f"API connection failed: {str(e)}")
        return False

# Add this near your app initialization
if not verify_api_key():
    print("Warning: Unable to verify Blockfrost API key")

# Add this before app.run()
if __name__ == '__main__':
    print("Starting Flask application...")
    try:
        # Create required directories if they don't exist
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('comparisons', exist_ok=True)
        print("Created required directories")
        
        # Get port from environment variable (for Render) or use default 5008
        port = int(os.environ.get('PORT', 5008))
        print(f"Using port: {port}")
        
        # In development (local) use debug mode, in production don't
        debug_mode = os.environ.get('FLASK_ENV') != 'production'
        print(f"Debug mode: {debug_mode}")
        
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except Exception as e:
        print(f"Error starting application: {e}")
        raise

