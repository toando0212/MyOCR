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

def predict_blocks(img):
    original = img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    linek = np.zeros((11, 11), dtype=np.uint8)
    linek[5, :] = 1
    x = cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek)
    gray = gray - x
    _, bw = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.dilate(bw, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_boxes = sort_boxes_top_to_bottom_left_to_right(bounding_boxes)
    sorted_contours = [contours[bounding_boxes.index(b)] for b in sorted_boxes]
    results = []
    block_index = 0
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        roi = original[y:y+h, x:x+w]
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
    results = predict_blocks(img)
    return jsonify({'results': results}), 200

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