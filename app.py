from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet import extract_face_embedding, cosine_similarity
from blink_detector import detect_blink_from_image
from firebase import mark_attendance
import os
import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

REGISTERED_EMBEDDINGS_PATH = 'embeddings.json'
TEMP_DIR = 'temp'

def load_registered_embeddings():
    if os.path.exists(REGISTERED_EMBEDDINGS_PATH):
        with open(REGISTERED_EMBEDDINGS_PATH, 'r') as f:
            return json.load(f)
    return {}

@app.route('/')
def home():
    return 'System is Running'

@app.route('/register', methods=['POST'])
def register_user():
    user_id = request.form.get('user_id')
    image = request.files.get('image')

    if not user_id or not image:
        return jsonify({"error": "Missing user_id or image"}), 400

    path = os.path.join(TEMP_DIR, f'{user_id}.jpg')
    image.save(path)

    embedding = extract_face_embedding(path)
    if embedding is not None:
        data = load_registered_embeddings()
        data[user_id] = embedding.tolist()
        with open(REGISTERED_EMBEDDINGS_PATH, 'w') as f:
            json.dump(data, f)
        return jsonify({"status": "registered", "user_id": user_id}), 200

    return jsonify({"error": "Face not detected"}), 400

@app.route('/mark_attendance', methods=['POST'])
def mark():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image provided"}), 400

    path = os.path.join(TEMP_DIR, 'mark.jpg')
    image.save(path)

    if not detect_blink_from_image(path):
        return jsonify({"error": "Make sure your eyes are closed"}), 403

    embedding = extract_face_embedding(path)
    if embedding is None:
        return jsonify({"error": "No face detected"}), 400

    data = load_registered_embeddings()
    for user_id, stored_embedding in data.items():
        sim = cosine_similarity(np.array(stored_embedding), embedding)
        if sim > 0.6:
            mark_attendance(user_id)
            return jsonify({
                "status": "Access granted",
                "user": user_id
            }), 200

    return jsonify({"error": "No matching face found"}), 404

@app.route('/recognize', methods=['POST'])
def recognize_user():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        base64_image = data['image'].split(',')[1]  # Remove header if exists
        img_data = base64.b64decode(base64_image)
        img = Image.open(BytesIO(img_data))
        img_path = os.path.join(TEMP_DIR, 'recognize.jpg')
        img.save(img_path)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    if not detect_blink_from_image(img_path):
        return jsonify({"error": "Make sure your eyes are closed"}), 403

    embedding = extract_face_embedding(img_path)
    if embedding is None:
        return jsonify({"error": "No face detected"}), 400

    data = load_registered_embeddings()
    for user_id, stored_embedding in data.items():
        sim = cosine_similarity(np.array(stored_embedding), embedding)
        if sim > 0.6:
            mark_attendance(user_id)
            return jsonify({
                "status": "access granted",
                "user": user_id
            }), 200

    return jsonify({"error": "No matching face found"}), 404

if __name__ == '__main__':
    os.makedirs(TEMP_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
