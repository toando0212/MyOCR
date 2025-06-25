from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_mysqldb import MySQL
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageOps
import io
import base64

# Import the OCR pipelines from their respective modules
# This keeps the main server file clean and modular.
from english_pipeline import ocr_pipeline as process_english
from viet_ocr import ocr_pipeline as process_vietnamese

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Server Environment Setup ---
# NOTE: Model device (CPU/GPU) is now handled within each pipeline module.
# Make sure your server environment has the correct PyTorch (GPU) version installed.
print("INFO: Server is starting. Model device configuration is managed by pipeline modules.")

# --- MySQL Configuration ---
# IMPORTANT: For production, update MYSQL_HOST to your database server's IP address.
app.config['MYSQL_HOST'] = '192.168.1.229'      # <-- CHANGE THIS FOR PRODUCTION
app.config['MYSQL_USER'] = 'myocr_user'     # Your MySQL username
app.config['MYSQL_PASSWORD'] = '0212'       # Your MySQL password
app.config['MYSQL_DB'] = 'myocr_db'         # Your MySQL database name
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mysql = MySQL(app)

# --- Model loading is now handled by the imported pipeline modules when they are first called ---
print("All models will be loaded by their respective pipeline modules upon first request.")


def encode_image_to_base64(pil_img):
    """Encodes a PIL image to a base64 string for JSON responses."""
    if pil_img is None:
        return None
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- API Endpoints ---

@app.route('/')
def health_check():
    """Health check endpoint to confirm the server is running."""
    return jsonify({'status': 'Flask backend is running.'})

@app.route('/classify', methods=['POST'])
def classify_blocks():
    """The main endpoint for uploading an image and receiving OCR results."""
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
        # Read image into memory for processing
        image_data = file.read()
        pil_img = Image.open(io.BytesIO(image_data))

        # Apply EXIF orientation correction to handle images from mobile devices
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # Ensure image is in RGB format for consistency
        pil_img = pil_img.convert("RGB")

        # Save the original image file for record-keeping
        filename = secure_filename(f"{user_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)

        # Create a database record for the uploaded image
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO images (user_id, image_path) VALUES (%s, %s)", (user_id, filepath))
        image_id = cur.lastrowid
        
        # Process the image using the appropriate OCR pipeline
        recognized_text = ""
        vis_img = None

        if language in ['vie', 'vi', 'vietnamese']:
            print("Running Vietnamese OCR pipeline via import...")
            # Unpack results: The Vietnamese pipeline might return multiple text versions
            _, recognized_text, _, _, _, vis_img = process_vietnamese(pil_img)
        else:
            print("Running English OCR pipeline via import...")
            # Unpack results: The English pipeline returns the final text directly
            recognized_text, _, _, _, vis_img = process_english(pil_img)

        vis_base64 = encode_image_to_base64(vis_img)

        # Store the recognized text in the database
        if image_id and recognized_text:
            cur.execute("INSERT INTO results (image_id, recognized_text) VALUES (%s, %s)", (image_id, recognized_text))
        
        mysql.connection.commit()

        # Format the response for the client
        api_results = [{'text': recognized_text}]

        print(f"Pipeline finished. Returning {len(recognized_text.splitlines())} lines of text.")
        return jsonify({'results': api_results, 'visualization': vis_base64}), 200

    except Exception as e:
        if cur:
            mysql.connection.rollback() # Rollback DB changes on error
        print(f"Error in /classify endpoint: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cur:
            cur.close()

@app.route('/register', methods=['POST'])
def register():
    """Endpoint for user registration."""
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
    """Endpoint for user login."""
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
    """Endpoint to retrieve a user's OCR history, grouped by session."""
    cur = None
    try:
        cur = mysql.connection.cursor()
        # Fetch all records, ensuring we get the image ID for deletion purposes
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

        # Group records into sessions based on a timeout
        sessions = []
        if history_records:
            current_session_records = [history_records[0]]
            SESSION_TIMEOUT_SECONDS = 30 # Time to consider uploads as part of one session

            for i in range(1, len(history_records)):
                prev_timestamp = current_session_records[-1][3]
                current_timestamp = history_records[i][3]
                
                if (current_timestamp - prev_timestamp).total_seconds() < SESSION_TIMEOUT_SECONDS:
                    current_session_records.append(history_records[i])
                else:
                    sessions.append(current_session_records)
                    current_session_records = [history_records[i]]
            sessions.append(current_session_records)

        # Format sessions for the JSON response
        history_list = []
        temp_session_id_counter = 0
        for session_records in reversed(sessions): # Show newest first
            session_details = []
            session_image_ids = []
            
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
                    'session_id': temp_session_id_counter,
                    'image_ids': session_image_ids, # Pass the real DB IDs for deletion
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
    """Endpoint to delete an entire history session based on its image IDs."""
    data = request.get_json()
    image_ids_to_delete = data.get('image_ids')

    if not image_ids_to_delete or not isinstance(image_ids_to_delete, list):
        return jsonify({'error': 'Invalid request. "image_ids" must be a list.'}), 400

    cur = None
    try:
        cur = mysql.connection.cursor()
        
        image_ids_to_delete = [int(id) for id in image_ids_to_delete]
        placeholders = ','.join(['%s'] * len(image_ids_to_delete))
        
        # Delete from child table first to respect foreign key constraints
        cur.execute(f"DELETE FROM results WHERE image_id IN ({placeholders})", image_ids_to_delete)
        
        # Then delete from the parent table
        cur.execute(f"DELETE FROM images WHERE id IN ({placeholders})", image_ids_to_delete)
        
        mysql.connection.commit()
        
        # Note: Image files on disk are not deleted here.
        # This prevents accidental data loss and can be handled by a separate cleanup script.

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
    # For production, it's recommended to use a WSGI server like Gunicorn:
    # Example: gunicorn --workers 4 --threads 2 --bind 0.0.0.0:5000 app_server:app
    #
    # The command below is for simple testing or development.
    # Host '0.0.0.0' makes the server accessible from other machines on the network.
    # Debug mode should be False in a production environment.
    app.run(host='0.0.0.0', port=5000, debug=False) 