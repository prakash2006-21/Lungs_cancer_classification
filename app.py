from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model("vgg19_lung_cancer.h5")
IMG_SIZE = 224

# Class labels
categories = [
    "Bengin cases",
    "Malignant cases",
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
    "normal"
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Read & preprocess image
            img = cv2.imread(file_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Prediction
            pred = model.predict(img)
            label = categories[np.argmax(pred)]
            confidence = np.max(pred) * 100

            return render_template("result.html", image_path=file_path, label=label, confidence=confidence)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
