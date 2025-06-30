from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_mysqldb import MySQL
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import argparse
import phunspell
import re
# from spellchecker import SpellChecker

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

# Khởi tạo phunspell cho tiếng Việt (chỉ khởi tạo 1 lần)
try:
    viet_spellchecker = phunspell.Phunspell('vi_VN')
    print("Vietnamese spellchecker initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Could not initialize Vietnamese spellchecker: {e}")
    viet_spellchecker = None

# Khởi tạo phunspell cho tiếng Anh
try:
    eng_spellchecker = phunspell.Phunspell('en_US')
    print("English spellchecker initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Could not initialize English spellchecker: {e}")
    eng_spellchecker = None

@app.route('/')
def health_check():
    return jsonify({'status': 'Flask backend is running.'})

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

        sessions = []
        if history_records:
            current_session_records = [history_records[0]]
            SESSION_TIMEOUT_SECONDS = 30

            for i in range(1, len(history_records)):
                prev_timestamp = current_session_records[-1][3]
                current_timestamp = history_records[i][3]
                
                if (current_timestamp - prev_timestamp).total_seconds() < SESSION_TIMEOUT_SECONDS:
                    current_session_records.append(history_records[i])
                else:
                    sessions.append(current_session_records)
                    current_session_records = [history_records[i]]
            sessions.append(current_session_records)

        history_list = []
        temp_session_id_counter = 0
        for session_records in reversed(sessions):
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
                    'image_ids': session_image_ids,
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
        
        image_ids_to_delete = [int(id) for id in image_ids_to_delete]
        placeholders = ','.join(['%s'] * len(image_ids_to_delete))
        
        sql_delete_results = f"DELETE FROM results WHERE image_id IN ({placeholders})"
        cur.execute(sql_delete_results, image_ids_to_delete)
        
        sql_delete_images = f"DELETE FROM images WHERE id IN ({placeholders})"
        cur.execute(sql_delete_images, image_ids_to_delete)
        
        mysql.connection.commit()
        
        return jsonify({'message': f'Successfully deleted session with {len(image_ids_to_delete)} images.'}), 200

    except Exception as e:
        if cur:
            mysql.connection.rollback()
        print(f"Error in /history/delete endpoint: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if cur:
            cur.close()

@app.route('/add_history', methods=['POST'])
def add_history():
    user_id = request.form.get('user_id')
    recognized_text = request.form.get('recognized_text')
    file = request.files.get('image')
    if not user_id or not recognized_text or not file:
        return jsonify({'error': 'Missing data'}), 400

    filename = secure_filename(f'{user_id}_{file.filename}')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO images (user_id, image_path) VALUES (%s, %s)", (user_id, filepath))
    image_id = cur.lastrowid
    cur.execute("INSERT INTO results (image_id, recognized_text) VALUES (%s, %s)", (image_id, recognized_text))
    mysql.connection.commit()
    cur.close()
    return jsonify({'message': 'History saved successfully'}), 200

@app.route('/spellcheck', methods=['POST'])
def spellcheck():
    data = request.get_json()
    text = data.get('text', '')
    language = data.get('language', 'vi_VN')

    print(f"[spellcheck] Nhận request với text: {text[:100]}... (length={len(text)}) và language: {language}")

    if not text:
        print("[spellcheck] Không có text gửi lên!")
        return jsonify({'error': 'No text provided'}), 400
    
    spellchecker = None
    if language in ['en', 'en_US', 'en-US', 'english']:
        if eng_spellchecker is None:
            print("[spellcheck] English spellchecker chưa sẵn sàng!")
            return jsonify({'error': 'English spellchecker is not available on the server.'}), 500
        spellchecker = eng_spellchecker
        language = 'en'
    elif language in ['vi', 'vi_vn', 'vi-VN', 'vietnamese', 'vie', 'vi_VN']:
        if viet_spellchecker is None:
            print("[spellcheck] Vietnamese spellchecker chưa sẵn sàng!")
            return jsonify({'error': 'Vietnamese spellchecker is not available on the server.'}), 500
        spellchecker = viet_spellchecker
        language = 'vi'
    else:
        print(f"[spellcheck] Ngôn ngữ không hỗ trợ: {language}")
        return jsonify({'error': 'Language not supported.'}), 400

    try:
        typos = []
        # Use re.finditer to get words and their positions. [\w'-]+ matches words with letters, numbers, underscore, hyphen, or apostrophe.
        for match in re.finditer(r"[\w'-]+", text):
            word = match.group(0)
            
            # Check spelling of the word
            if not spellchecker.lookup(word):
                suggestions = list(spellchecker.suggest(word))
                # Only consider it a typo if there are suggestions
                if suggestions:
                    start, end = match.span()
                    typos.append({
                        'word': word,
                        'start': start,
                        'end': end,
                        'suggestions': suggestions
                    })
        
        # Generate corrected_text by replacing typos in the original text
        # Process replacements from the end to avoid messing up indices of unprocessed typos.
        corrected_text_list = list(text)
        for typo in sorted(typos, key=lambda x: x['start'], reverse=True):
            suggestions = typo['suggestions']
            if suggestions:
                start = typo['start']
                end = typo['end']
                replacement = suggestions[0]
                corrected_text_list[start:end] = list(replacement)
        
        corrected_text = "".join(corrected_text_list)

        print(f"[spellcheck] Hoàn thành. Số từ sửa: {len(typos)}. Trả kết quả về client với thông tin chi tiết.")
        return jsonify({
            'corrected_text': corrected_text,
            'typos': typos
        }), 200
    except Exception as e:
        print(f"Error during spellcheck: {e}")
        return jsonify({'error': f'An error occurred during spellcheck: {str(e)}'}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Flask app on a specified port')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', debug=True, port=args.port)