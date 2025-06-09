from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_mysqldb import MySQL
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from joblib import load
import pandas as pd
import io
from werkzeug.security import generate_password_hash, check_password_hash
import pytesseract
from PIL import Image, ImageEnhance
try:
    from deskew import determine_skew
except ImportError:
    determine_skew = None  # If deskew is not installed, skip deskewing
import time

# Global constants
TESSDATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Tesseract-OCR', 'tessdata')

# Set pytesseract to use the local Tesseract-OCR binary
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.dirname(__file__), '..', 'Tesseract-OCR', 'tesseract.exe')
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR'  # Not needed when using --tessdata-dir

app = Flask(__name__)
CORS(app)

# MySQL configuration (update with your credentials)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'myocr_user'
app.config['MYSQL_PASSWORD'] = '0212'
app.config['MYSQL_DB'] = 'myocr_db'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mysql = MySQL(app)

# Load model and label file
MODEL_PATH = '../Handwritten-and-Printed-Text-Classification-in-Doctors-Prescription/Handwritten-and-Printed-Text-Classification-in-Doctors-Prescription/full_model.joblib'
LABEL_PATH = '../res2.csv'
model = load(MODEL_PATH)
labels = pd.read_csv(LABEL_PATH, header=None)[0].tolist()

def extract_features(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = img.shape[0]
    cols = img.shape[1]
    arr = [rows, cols, rows / cols if cols > 0 else 0]
    retval, bwMask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    myavg = 0
    for xx in range(cols):
        mycnt = 0
        for yy in range(rows):
            if bwMask[yy, xx] == 0:
                mycnt += 1
        myavg += (mycnt * 1.0) / rows
    myavg /= cols if cols > 0 else 1
    arr.append(myavg)
    change = 0
    for xx in range(rows):
        mycnt = 0
        for yy in range(cols - 1):
            if (bwMask[xx, yy] == 0) != (bwMask[xx, yy + 1] == 0):
                mycnt += 1
        change += (mycnt * 1.0) / cols if cols > 0 else 1
    change /= rows if rows > 0 else 1
    arr.append(change)
    return arr

def group_into_lines(boxes, y_threshold=10):
    lines = []
    for box in sorted(boxes, key=lambda b: b[1]):  # sort by top y
        x, y, w, h = box
        placed = False
        for line in lines:
            _, ly, _, lh = line[0]
            if abs(y - ly) < y_threshold:  # same line
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])
    return lines

def sort_boxes_top_to_bottom_left_to_right(boxes):
    lines = group_into_lines(boxes)
    sorted_boxes = []
    for line in lines:
        sorted_line = sorted(line, key=lambda b: b[0])  # sort by x
        sorted_boxes.extend(sorted_line)
    return sorted_boxes

def preprocess_for_ocr(img):
    """
    This function now matches the notebook's pre_process_image logic exactly:
    - Downscale by 0.3x
    - Convert BGR to RGB, then to grayscale
    - Adaptive thresholding with (5, 11)
    """
    # 1. Downscale by 0.3x
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    # 2. Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 4. Adaptive thresholding (blockSize=5, C=11)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 11)
    return img

def get_tesseract_config(label, lang='eng'):
    """Get optimized Tesseract configuration based on content type"""
    base_config = f'--tessdata-dir {TESSDATA_DIR}'
    
    if label == 'Handwritten_extended':
        return f'{base_config} --psm 6 --oem 1 -c tessedit_char_blacklist=|#<>_{{}} -c textord_heavy_nr=1 -c textord_noise_rejrows=1'
    elif label == 'Printed_extended':
        return f'{base_config} --psm 3 --oem 1 -c preserve_interword_spaces=1 -c tessedit_do_invert=0'
    else:
        return f'{base_config} --psm 4 --oem 1'

def predict_blocks(img, language):
    """
    Detects text blocks using a safer preprocessing method, classifies them,
    and returns sorted bounding boxes.
    """
    original = img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Use a simpler, less destructive preprocessing for contour detection.
    # The goal here is just to find the bounding boxes accurately.
    # The heavy image enhancement for OCR is done later in ocr_region.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 4
    )

    # Use dilation to connect characters into words and words into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
    dilated = cv2.dilate(bw, kernel, iterations=3)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter based on area to remove non-text objects
        if w > 10 and h > 10:
             bounding_boxes.append((x,y,w,h))
    
    # Sort boxes from top to bottom, then left to right
    sorted_boxes = sort_boxes_top_to_bottom_left_to_right(bounding_boxes)
    
    results = []
    block_index = 0
    for (x, y, w, h) in sorted_boxes:
        roi = original[y:y+h, x:x+w]
        
        # Skip if ROI is empty
        if roi.size == 0:
            continue

        feat = extract_features(roi)
        pred = model.predict([feat])[0]
        
        try:
            pred_idx = int(pred)
            label = labels[pred_idx] if pred_idx < len(labels) else str(pred)
        except (ValueError, TypeError):
            label = str(pred)
            
        results.append({
            'block_index': block_index,
            'label': label,
            'box': [int(x), int(y), int(w), int(h)]
        })
        block_index += 1
        
    return results

def ocr_region(region_img, label, lang='eng'):
    """Enhanced OCR with better preprocessing and configuration"""
    # Get optimal configuration
    tessdata_config = get_tesseract_config(label, lang)

    # Use the new, specific preprocessing for handwritten/mixed classes
    if label in ['Handwritten_extended', 'Mixed_extended']:
        pre_img = preprocess_for_ocr(region_img)
        pil_img = Image.fromarray(pre_img)
    else:
        # For printed text, a simpler preprocessing is sufficient
        if len(region_img.shape) == 3:
            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = region_img.copy()
        
        # Rescale for consistency
        h, w = gray.shape
        if h < 100: # A smaller threshold for printed text
             scale_factor = 200 / h
             gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        pil_img = Image.fromarray(gray)
        
    # Perform OCR
    try:
        text = pytesseract.image_to_string(pil_img, lang=lang, config=tessdata_config)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""

def detect_text_regions_tesseract(img, psm=1):
    """
    Use Tesseract's layout analysis to detect text regions (bounding boxes) only.
    Returns a list of dicts: {left, top, width, height}
    """
    custom_config = f'--psm {psm} --oem 1'
    data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        # Only keep boxes with some text detected (block/line/word)
        if int(data['width'][i]) > 0 and int(data['height'][i]) > 0:
            boxes.append({
                'left': int(data['left'][i]),
                'top': int(data['top'][i]),
                'width': int(data['width'][i]),
                'height': int(data['height'][i])
            })
    return boxes


def classify_region_rf(roi_img):
    """
    Extract features from ROI and classify using the trained RF model.
    Returns 'Handwritten' or 'Printed'.
    """
    features = extract_features(roi_img)
    pred = model.predict([features])[0]
    try:
        pred_idx = int(pred)
        label = labels[pred_idx] if pred_idx < len(labels) else str(pred)
    except (ValueError, TypeError):
        label = str(pred)
    return label


def preprocess_handwritten_region(roi_img):
    """
    Custom preprocessing for handwritten regions: contrast, denoise, deskew.
    """
    # Contrast enhancement
    pil_img = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(2.0)
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Deskew
    if determine_skew is not None:
        angle = determine_skew(gray)
        if angle is not None and abs(angle) < 45:
            (h, w) = gray.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=30, templateWindowSize=7, searchWindowSize=21)
    # Adaptive threshold
    binarized = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )
    return binarized


def pipeline_document_image(img, psm=1, lang='eng'):
    """
    Full pipeline: detect regions, classify, OCR with appropriate preprocessing.
    Returns a list of dicts: {box, label, text}
    """
    results = []
    boxes = detect_text_regions_tesseract(img, psm=psm)
    for box in boxes:
        x, y, w, h = box['left'], box['top'], box['width'], box['height']
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        label = classify_region_rf(roi)
        if label.lower().startswith('handwritten'):
            proc_img = preprocess_handwritten_region(roi)
            pil_img = Image.fromarray(proc_img)
            tess_config = '--psm 6 --oem 1'
        else:
            # Printed: simple grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
            pil_img = Image.fromarray(gray)
            tess_config = '--psm 3 --oem 1'
        try:
            text = pytesseract.image_to_string(pil_img, lang=lang, config=tess_config)
        except Exception as e:
            text = ''
        results.append({
            'box': [x, y, w, h],
            'label': label,
            'text': text.strip()
        })
    return results

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
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    language = request.form.get('language', 'en')  # default to English
    t0 = time.time()
    results = predict_blocks(img, language)
    t1 = time.time()
    print(f"Block detection/classification took {t1 - t0:.2f} seconds")
    print(f"Number of regions detected: {len(results)}")
    # For each region, run OCR and add recognized text
    ocr_results = []
    t2 = time.time()
    for region in results:
        x, y, w, h = region['box']
        roi = img[y:y+h, x:x+w]
        t_start = time.time()
        text = ocr_region(roi, region['label'], lang=language)
        print(f"OCR for region {region['block_index']} took {time.time() - t_start:.2f} seconds")
        ocr_results.append({
            'block_index': region['block_index'],
            'label': region['label'],
            'box': region['box'],
            'text': text
        })
    t3 = time.time()
    print(f"Total OCR time: {t3 - t2:.2f} seconds")
    print(f"Total /classify endpoint time: {t3 - t0:.2f} seconds")

    # Store recognized text in the results table
    # Try to find the image in the images table by filename
    filename = file.filename
    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM images WHERE image_path LIKE %s ORDER BY uploaded_at DESC LIMIT 1", (f"%{filename}",))
    row = cur.fetchone()
    if row:
        image_id = row[0]
        recognized_text = "".join([block['text'] for block in ocr_results])
        cur.execute("INSERT INTO results (image_id, recognized_text) VALUES (%s, %s)", (image_id, recognized_text))
        mysql.connection.commit()
    cur.close()

    return jsonify({'results': ocr_results}), 200

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

@app.route('/test_tess', methods=['POST'])
def test_tess():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tess_config = f'--tessdata-dir {TESSDATA_DIR} --psm 3 --oem 1'
    try:
        text = pytesseract.image_to_string(pil_img, config=tess_config)
    except Exception as e:
        return jsonify({'error': f'OCR error: {str(e)}'}), 500
    return jsonify({'text': text.strip()}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')