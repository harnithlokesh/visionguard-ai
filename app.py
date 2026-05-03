from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import tensorflow as tf

from main import highlight_differences, DistanceLayer

app = Flask(__name__)
CORS(app)  # allow React to talk to Flask

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# Load Model
# ==============================
model = tf.keras.models.load_model(
    'image_comparison_model.keras',
    custom_objects={'DistanceLayer': DistanceLayer},
    compile=False
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# Serve Uploaded Images
# ==============================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ==============================
# Compare Route (API)
# ==============================
@app.route('/compare', methods=['POST'])
def compare():
    try:
        # Get uploaded files
        original = request.files['original']
        edited = request.files['edited']

        # Save files
        original_path = os.path.join(UPLOAD_FOLDER, original.filename)
        edited_path = os.path.join(UPLOAD_FOLDER, edited.filename)

        original.save(original_path)
        edited.save(edited_path)

        # ==============================
        # Highlight Differences (OpenCV)
        # ==============================
        output_filename = 'output.jpg'
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

        # ==============================
        # Return JSON (for React)
        # ==============================
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
# Run App
# ==============================
if __name__ == '__main__':
    app.run(debug=True)