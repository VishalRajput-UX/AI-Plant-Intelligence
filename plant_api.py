from flask_cors import CORS
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ==========================
# Flask App Setup
# ==========================
app = Flask(__name__)
CORS(app)

# ==========================
# Load Model + Labels
# ==========================
img_size = (224,224)

model = tf.keras.models.load_model("plant_health_model.h5")
species_names = np.load("species_names.npy")

print("Model Loaded Successfully ✅")

# ==========================
# Image Preprocessing
# ==========================
def preprocess(image):

    image = image.resize(img_size)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    return arr

# ==========================
# Home Route
# ==========================
@app.route('/')
def home():
    return "Plant Detection ML API Running 🚀"

# ==========================
# Prediction Route
# ==========================
@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({"error":"No Image Uploaded"})

    file = request.files['image']

    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    arr = preprocess(img)

    plant_prob, health_prob = model.predict(arr)

    plant_idx = np.argmax(plant_prob[0])
    plant_name = species_names[plant_idx]

    confidence = float(np.max(plant_prob[0])) * 100
    health = "Healthy" if health_prob[0][0] > 0.5 else "Unhealthy"

    return jsonify({
        "plant": str(plant_name),
        "health": health,
        "confidence": round(confidence,2)
    })

# ==========================
# Run Server
# ==========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)