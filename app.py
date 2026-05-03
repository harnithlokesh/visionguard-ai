from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import tensorflow as tf
import gdown

from main import highlight_differences, DistanceLayer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# MODEL DOWNLOAD (Google Drive)
# ==============================
MODEL_PATH = "image_comparison_model.keras"

FILE_ID = "1uUuzbSWtKpFQwaWp9W1QnscZ_CjDaNri"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ==============================
# LOAD MODEL
# ==============================
print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'DistanceLayer': DistanceLayer},
    compile=False
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model loaded successfully.")

# ==============================
# SERVE UPLOADED FILES
# ==============================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ==============================
# COMPARE ROUTE
# ==============================
@app.route('/compare', methods=['POST'])
def compare():
    try:
        if 'original' not in request.files or 'edited' not in request.files:
            return jsonify({
                "success": False,
                "error": "Both images are required"
            })

        original = request.files['original']
        edited = request.files['edited']

        original_path = os.path.join(UPLOAD_FOLDER, original.filename)
        edited_path = os.path.join(UPLOAD_FOLDER, edited.filename)

        original.save(original_path)
        edited.save(edited_path)

        # ==============================
        # Highlight Differences
        # ==============================
        output_filename = f"output_{original.filename}"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)

        highlight_differences(original_path, edited_path, output_path)

        # ==============================
        # ML Prediction
        # ==============================
        img_size = (128, 128)

        orig_img = cv2.resize(cv2.imread(original_path), img_size) / 255.0
        edit_img = cv2.resize(cv2.imread(edited_path), img_size) / 255.0

        prediction = model.predict(
            [np.array([orig_img]), np.array([edit_img])]
        )

        similarity = 1 - prediction[0][0]
        similarity_percent = float(similarity * 100)

        verdict = "Edited" if similarity < 0.5 else "Original"

        return jsonify({
            "success": True,
            "similarity": round(similarity_percent, 2),
            "verdict": verdict,
            "image_path": f"/uploads/{output_filename}"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# ==============================
# HEALTH CHECK (for Render)
# ==============================
@app.route('/')
def home():
    return jsonify({"status": "Backend is running"})

# ==============================
# RUN APP
# ==============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)