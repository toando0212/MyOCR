from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_mysqldb import MySQL
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from joblib import load
import io
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageDraw, ImageFont
import time
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import math
import base64

# Global constants
app = Flask(__name__)
CORS(app)

# --- Server Environment Setup ---
# Check for GPU and set the device for PyTorch models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO: Using device: {DEVICE}")

# Font for visualization - place arial.ttf in the same directory as app.py
FONT_PATH = os.path.join(os.path.dirname(__file__), "arial.ttf")
if not os.path.exists(FONT_PATH):
    print(f"WARNING: Font not found at {FONT_PATH}. Visualization may use a default font.")


# MySQL configuration - IMPORTANT: Update these for your server's database
app.config['MYSQL_HOST'] = 'localhost'      # Or the IP of your MySQL server
app.config['MYSQL_USER'] = 'myocr_user'     # Your MySQL username
app.config['MYSQL_PASSWORD'] = '0212'       # Your MySQL password
app.config['MYSQL_DB'] = 'myocr_db'         # Your MySQL database name
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mysql = MySQL(app)

# --- Model Loading ---
print("Loading models...")
# Classifier for handwritten/printed text
MODEL_PATH = 'full_model.joblib'
try:
    classifier_model = load(MODEL_PATH)
    print("Classifier model loaded.")
except Exception as e:
    print(f"Could not load classifier model: {e}")
    classifier_model = None

# Doctr for line segmentation
det_model = detection_predictor(arch="db_mobilenet_v3_large", pretrained=True)
print("DocTR detection model loaded.")

# VietOCR for Vietnamese text recognition
vietocr_config = Cfg.load_config_from_name('vgg_seq2seq')
vietocr_config['device'] = DEVICE # Use 'cuda' or 'cpu'
vietocr_config['predictor']['beamsearch'] = False
vietocr_predictor = Predictor(vietocr_config)
print("VietOCR model loaded.")

# TrOCR for English text recognition
printed_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
printed_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
printed_model.to(DEVICE) # Move model to GPU if available
print("TrOCR Printed model loaded.")

handwritten_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
handwritten_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
handwritten_model.to(DEVICE) # Move model to GPU if available
print("TrOCR Handwritten model loaded.")
print("All models loaded successfully.")


def extract_features(img):
    """Extracts features for the classifier from a given image region."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    if cols == 0 or rows == 0:
        return [0, 0, 0, 0, 0]

    arr = [rows, cols, rows / cols if cols > 0 else 0]
    
    _, bwMask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    myavg = 0
    if cols > 0:
        for xx in range(cols):
            mycnt = np.sum(bwMask[:, xx] == 0)
            myavg += (mycnt * 1.0) / rows
        myavg /= cols
    arr.append(myavg)
    
    change = 0
    if rows > 0:
        for xx in range(rows):
            row_data = bwMask[xx, :]
            mycnt = np.sum(row_data[:-1] != row_data[1:])
            change += (mycnt * 1.0) / (cols if cols > 0 else 1)
        change /= rows
    arr.append(change)
    
    return arr

def classify_region(roi_img):
    """Classifies an image region as 'Handwritten' or 'Printed'."""
    if classifier_model is None:
        return "Printed" # Default if model is not available
    try:
        features = extract_features(np.array(roi_img))
        pred = classifier_model.predict([features])[0]
        return str(pred)
    except Exception as e:
        print(f"Classification failed: {e}")
        return "Unknown"

def deskew_image(pil_img: Image.Image) -> Image.Image:
    """Detects the skew angle of the text in the image and rotates it to be straight."""
    try:
        img = np.array(pil_img.convert("L"))
        img_inverted = cv2.bitwise_not(img)
        thresh = cv2.threshold(img_inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return pil_img

        angles = [math.degrees(math.atan2(y2 - y1, x2 - x1)) for line in lines for x1, y1, x2, y2 in line]
        median_angle = np.median(angles)

        if abs(median_angle) < 1:
            return pil_img

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    except Exception as e:
        print(f"Error during image deskewing: {e}")
        return pil_img

def remove_horizontal_lines(pil_img):
    try:
        img = np.array(pil_img)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.mean() > 127:
            img = 255 - img
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        img_no_lines = cv2.subtract(img, detected_lines)
        img_no_lines = 255 - img_no_lines
        return Image.fromarray(img_no_lines)
    except Exception as e:
        print(f"Error in horizontal line removal: {e}")
        return pil_img

def doctr_segment_lines(pil_img):
    """Segments an image into lines of text using DocTR."""
    temp_path = None
    try:
        temp_path = "temp_input_doctr.png"
        pil_img.convert("RGB").save(temp_path)
        
        doc = DocumentFile.from_images(temp_path)
        result = det_model(doc)
        
        if not result or 'words' not in result[0] or result[0]['words'].shape[0] == 0:
            return []

        page = result[0]
        img_width, img_height = pil_img.size
        words = [
            (int(w[0] * img_width), int(w[1] * img_height), int(w[2] * img_width), int(w[3] * img_height))
            for w in page['words'][:, :-1]
        ]
        words.sort(key=lambda w: (w[1], w[0]))

        if not words:
            return []

        lines = []
        current_line = [words[0]]
        for box in words[1:]:
            last_box = current_line[-1]
            last_box_y_center = (last_box[1] + last_box[3]) / 2
            current_box_y_center = (box[1] + box[3]) / 2
            last_box_height = last_box[3] - last_box[1]

            if abs(current_box_y_center - last_box_y_center) < last_box_height * 0.7:
                current_line.append(box)
            else:
                min_x = min(b[0] for b in current_line)
                min_y = min(b[1] for b in current_line)
                max_x = max(b[2] for b in current_line)
                max_y = max(b[3] for b in current_line)
                lines.append((min_x, min_y, max_x, max_y))
                current_line = [box]
        
        if current_line:
            min_x = min(b[0] for b in current_line)
            min_y = min(b[1] for b in current_line)
            max_x = max(b[2] for b in current_line)
            max_y = max(b[3] for b in current_line)
            lines.append((min_x, min_y, max_x, max_y))

        return lines
    except Exception as e:
        print(f"Error in line segmentation: {e}")
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@torch.no_grad()
def run_english_pipeline(pil_img):
    """Full English OCR pipeline."""
    # 1. Preprocessing
    deskewed_img = deskew_image(pil_img)
    processed_img = remove_horizontal_lines(deskewed_img)
    
    # 2. Line Detection
    line_boxes = doctr_segment_lines(processed_img)
    if not line_boxes:
        return [], pil_img

    # 3. Recognition per line
    ocr_results = []
    vis_img = deskewed_img.copy().convert("RGB")
    draw = ImageDraw.Draw(vis_img)
    try:
        font = ImageFont.truetype(FONT_PATH, 15)
    except IOError:
        print(f"Could not load font {FONT_PATH}, using default font.")
        font = ImageFont.load_default()

    for i, box in enumerate(line_boxes):
        x_min, y_min, x_max, y_max = box
        crop = processed_img.crop((x_min, y_min, x_max, y_max))
        
        if np.array(crop).std() < 10:
            continue

        line_label = classify_region(crop.convert("RGB"))
        
        if 'handwritten' in line_label.lower():
            processor, model = handwritten_processor, handwritten_model
        else:
            processor, model = printed_processor, printed_model
        
        pixel_values = processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values.to(DEVICE)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        if len(text) < 2:
            continue

        ocr_results.append({'block_index': i, 'box': box, 'label': line_label, 'text': text})
        
        color = "green" if 'handwritten' in line_label.lower() else "blue"
        draw.rectangle(box, outline=color, width=2)
        display_text = f"[{line_label}] {text}"
        text_position = (box[0], box[1] - 15 if box[1] > 15 else box[1])
        draw.text(text_position, display_text, fill=color, font=font)
        
    return ocr_results, vis_img

@torch.no_grad()
def run_vietnamese_pipeline(pil_img):
    """Full Vietnamese OCR pipeline."""
    # 1. Preprocessing
    deskewed_img = deskew_image(pil_img)
    processed_img = remove_horizontal_lines(deskewed_img)
    
    # 2. Line Detection
    line_boxes = doctr_segment_lines(processed_img)
    if not line_boxes:
        return [], pil_img

    # 3. Recognition per line
    ocr_results = []
    vis_img = deskewed_img.copy().convert("RGB")
    draw = ImageDraw.Draw(vis_img)

    for i, box in enumerate(line_boxes):
        x_min, y_min, x_max, y_max = box
        crop = processed_img.crop((x_min, y_min, x_max, y_max))
        
        if np.array(crop).std() < 10:
            continue
            
        text = vietocr_predictor.predict(crop)
        
        if len(text) < 2:
            continue
            
        ocr_results.append({'block_index': i, 'box': box, 'label': 'Vietnamese', 'text': text})
        
        draw.rectangle(box, outline="orange", width=2)
        text_position = (box[0], box[1] - 15 if box[1] > 15 else box[1])
        draw.text(text_position, text, fill="orange")
        
    return ocr_results, vis_img

def encode_image_to_base64(pil_img):
    """Encodes a PIL image to a base64 string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def health_check():
    return jsonify({'status': 'Flask backend is running.'})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # Get user_id from form
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'No user_id provided'}), 400
    # Store filepath in MySQL
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO images (user_id, image_path) VALUES (%s, %s)", (user_id, filepath))
    mysql.connection.commit()
    cur.close()
    return jsonify({'message': 'Image uploaded successfully', 'path': filepath}), 201

@app.route('/classify', methods=['POST'])
def classify_blocks():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img_cv = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_cv is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Convert to PIL Image for pipelines
        pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        language = request.form.get('language', 'eng').lower()

        if language in ['vie', 'vi', 'vietnamese']:
            print("Running Vietnamese OCR pipeline...")
            ocr_results, vis_img = run_vietnamese_pipeline(pil_img)
        else:
            print("Running English OCR pipeline...")
            ocr_results, vis_img = run_english_pipeline(pil_img)

        # Encode visualization image to base64
        vis_base64 = encode_image_to_base64(vis_img)

        # Store results in DB
        filename = secure_filename(file.filename)
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM images WHERE image_path LIKE %s ORDER BY uploaded_at DESC LIMIT 1", (f"%{filename}",))
        row = cur.fetchone()
        if row:
            image_id = row[0]
            recognized_text = "\n".join([block['text'] for block in ocr_results])
            cur.execute("INSERT INTO results (image_id, recognized_text) VALUES (%s, %s)", (image_id, recognized_text))
            mysql.connection.commit()
        cur.close()

        print(f"Pipeline finished. Found {len(ocr_results)} text blocks.")
        return jsonify({'results': ocr_results, 'visualization': vis_base64}), 200

    except Exception as e:
        print(f"Error in /classify endpoint: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 