from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_mysqldb import MySQL
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageOps
import io
import base64

# Import the OCR pipelines
# These files now contain all the model loading and processing logic.
from english_pipeline import ocr_pipeline as process_english
from viet_ocr import ocr_pipeline as process_vietnamese

# Global constants
app = Flask(__name__)
CORS(app)

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'myocr_user'
app.config['MYSQL_PASSWORD'] = '0212'
app.config['MYSQL_DB'] = 'myocr_db'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mysql = MySQL(app)

# --- Model Loading is now handled within the imported pipeline files ---
print("All models are being loaded by their respective pipeline modules...")


def encode_image_to_base64(pil_img):
    """Encodes a PIL image to a base64 string."""
    if pil_img is None:
        return None
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def health_check():
    return jsonify({'status': 'Flask backend is running.'})

@app.route('/classify', methods=['POST'])
def classify_blocks():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    user_id = request.form.get('user_id')
    language = request.form.get('language', 'eng').lower()

    if not user_id:
        return jsonify({'error': 'No user_id provided'}), 400

    cur = None
    try:
        # --- Read image into memory first for processing ---
        image_data = file.read()
        pil_img = Image.open(io.BytesIO(image_data))

        # --- FIX: Apply EXIF orientation correction ---
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # --- Now convert to RGB after orientation is fixed ---
        pil_img = pil_img.convert("RGB")

        # --- Now, save the original image data to a file for record-keeping ---
        filename = secure_filename(f"{user_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO images (user_id, image_path) VALUES (%s, %s)", (user_id, filepath))
        image_id = cur.lastrowid
        
        # --- Process the image for OCR using imported pipelines (pil_img is already loaded) ---
        recognized_text = ""
        vis_img = None

        if language in ['vie', 'vi', 'vietnamese']:
            print("Running Vietnamese OCR pipeline via import...")
            # Unpack the results from the Vietnamese pipeline
            # The second element is the post-processed text we want
            raw_text, post_processed_text, _, _, _, vis_img = process_vietnamese(pil_img)
            recognized_text = post_processed_text
        else:
            print("Running English OCR pipeline via import...")
            # Unpack the results from the English pipeline
            # The first element is the text we want
            raw_text, _, _, _, vis_img = process_english(pil_img)
            recognized_text = raw_text

        vis_base64 = encode_image_to_base64(vis_img)

        # --- Store OCR results in DB ---
        if image_id and recognized_text:
            cur.execute("INSERT INTO results (image_id, recognized_text) VALUES (%s, %s)", (image_id, recognized_text))
        
        mysql.connection.commit()

        # --- Format the response for the Android client ---
        # The client expects a 'results' array of objects, with each object having a 'text' key.
        # We'll create a single result object containing all the recognized text.
        api_results = [{'text': recognized_text}]

        print(f"Pipeline finished. Returning {len(recognized_text.splitlines())} lines of text.")
        return jsonify({'results': api_results, 'visualization': vis_base64}), 200

    except Exception as e:
        if cur:
            mysql.connection.rollback()
        print(f"Error in /classify endpoint: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cur:
            cur.close()

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if len(username) < 4 or len(password) < 6:
        return jsonify({'error': 'Invalid username or password'}), 400
    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    if cur.fetchone():
        cur.close()
        return jsonify({'error': 'Username already exists'}), 409
    hashed_pw = generate_password_hash(password)
    cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
    mysql.connection.commit()
    user_id = cur.lastrowid
    cur.close()
    return jsonify({'message': 'User registered successfully', 'user_id': user_id}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, password FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    cur.close()
    if row is None:
        return jsonify({'error': 'Invalid username or password'}), 401
    user_id, pw_hash = row
    if not check_password_hash(pw_hash, password):
        return jsonify({'error': 'Invalid username or password'}), 401
    return jsonify({'message': 'Login successful', 'user_id': user_id}), 200

@app.route('/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    cur = None
    try:
        cur = mysql.connection.cursor()
        # Fetch all records for the user, ordered by time
        # IMPORTANT: Fetch the image ID (i.id) to identify records for deletion
        query = """
        SELECT i.id, i.image_path, r.recognized_text, i.uploaded_at
        FROM images i
        JOIN results r ON i.id = r.image_id
        WHERE i.user_id = %s
        ORDER BY i.uploaded_at ASC
        """
        cur.execute(query, (user_id,))
        history_records = cur.fetchall()
        
        if not history_records:
            return jsonify([]), 200

        # Group records into sessions based on timestamp
        sessions = []
        if history_records:
            current_session_records = [history_records[0]]
            SESSION_TIMEOUT_SECONDS = 30  # Increased timeout for more robust session grouping

            for i in range(1, len(history_records)):
                prev_timestamp = current_session_records[-1][3] # Index 3 is uploaded_at
                current_timestamp = history_records[i][3]
                
                if (current_timestamp - prev_timestamp).total_seconds() < SESSION_TIMEOUT_SECONDS:
                    current_session_records.append(history_records[i])
                else:
                    sessions.append(current_session_records)
                    current_session_records = [history_records[i]]
            sessions.append(current_session_records) # Add the last session

        # Format the sessions for the response
        history_list = []
        # A simple counter to act as a session ID for this request
        temp_session_id_counter = 0
        for session_records in reversed(sessions): # Show newest sessions first
            session_details = []
            session_image_ids = [] # Collect image IDs for this session
            
            for record in session_records:
                image_id, image_path, recognized_text, _ = record
                session_image_ids.append(image_id)
                
                encoded_image = None
                if os.path.exists(image_path):
                    with open(image_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                if encoded_image and recognized_text:
                    session_details.append({
                        'image_base64': encoded_image,
                        'text': recognized_text
                    })
            
            if session_details:
                history_list.append({
                    'session_id': temp_session_id_counter, # a temporary ID for the client
                    'image_ids': session_image_ids, # The important part for deletion
                    'timestamp': session_records[0][3].strftime('%Y-%m-%d %H:%M:%S'),
                    'image_count': len(session_records),
                    'results': session_details
                })
                temp_session_id_counter += 1
            
        return jsonify(history_list), 200

    except Exception as e:
        print(f"Error in /history endpoint: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cur:
            cur.close()

@app.route('/history/delete', methods=['POST'])
def delete_history_session():
    data = request.get_json()
    image_ids_to_delete = data.get('image_ids')

    if not image_ids_to_delete or not isinstance(image_ids_to_delete, list):
        return jsonify({'error': 'Invalid request. "image_ids" must be a list.'}), 400

    cur = None
    try:
        cur = mysql.connection.cursor()
        
        # To be safe, ensure all IDs are integers
        image_ids_to_delete = [int(id) for id in image_ids_to_delete]
        
        # Create placeholders for the IN clause
        placeholders = ','.join(['%s'] * len(image_ids_to_delete))
        
        # Delete from results first (child table)
        sql_delete_results = f"DELETE FROM results WHERE image_id IN ({placeholders})"
        cur.execute(sql_delete_results, image_ids_to_delete)
        
        # Delete from images (parent table)
        sql_delete_images = f"DELETE FROM images WHERE id IN ({placeholders})"
        cur.execute(sql_delete_images, image_ids_to_delete)
        
        mysql.connection.commit()
        
        # Optionally, delete the image files from the server
        # This part is commented out as it requires fetching paths before deleting DB records.
        # It's safer to have a separate cleanup script for orphaned files.
        # for image_id in image_ids_to_delete:
        #    ... find path and os.remove(path) ...

        return jsonify({'message': f'Successfully deleted session with {len(image_ids_to_delete)} images.'}), 200

    except Exception as e:
        if cur:
            mysql.connection.rollback()
        print(f"Error in /history/delete endpoint: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cur:
            cur.close()

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')