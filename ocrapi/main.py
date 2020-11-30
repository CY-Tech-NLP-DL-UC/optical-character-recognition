import os
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request
from werkzeug.utils import secure_filename

from lp_prediction_pytorch.lp_prediction import get_prediction
from pl_detection import pl_detection

# from .pl_detection.pl_detection import pl_detection

# import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

UPLOAD_FOLDER = Path(__file__).resolve().parent / "static" / "uploads"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "static" / "outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY="dev",
    DATABASE=os.path.join(app.instance_path, "ocrapi.sqlite"),
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    OUTPUT_FOLDER=OUTPUT_FOLDER,
)

filename_logo = "TheApp.jpg"
rel_path_logo = Path("..") / "static" / filename_logo

@app.route("/gimme_your_plate", methods=["GET", "POST"])
def plate():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            original_path = app.config["UPLOAD_FOLDER"] / filename
            original_rel_path = Path("..") / "static" / "uploads" / filename
            file.save(original_path)
            filename = "extracted_plate.jpg"
            plate_path = os.path.join(os.path.dirname(original_path), "..", "outputs", filename)
            if os.path.isfile(plate_path):
                os.remove(plate_path)
            image_array = pl_detection.main(original_path)
            im = Image.fromarray((image_array * 255).astype(np.uint8))
            print(plate_path)
            im.save(plate_path)
            input_img = open(plate_path, "rb")
            img_bytes = input_img.read()
            response = get_prediction(image_bytes=img_bytes)
            plate_rel_path = Path("..") / "static" / "outputs" / filename
        return render_template(
            "results_plate_detection.html",
            logo=rel_path_logo,
            original_img=original_rel_path,
            plate_img=plate_rel_path,
            response=response,
        )
    return render_template("license_plate.html",
            logo=rel_path_logo)

@app.route("/gimme_your_letter")
def letter():
    return render_template("handwritten.html")

@app.route("/")
def home():
    filename_logo = "TheApp.jpg"
    rel_path_logo = Path("..") / "static" / filename_logo
    return render_template("home.html", logo=rel_path_logo)

@app.route("/welcome")
def welcome():
    return "Welcome!"

@app.route("/recognize_text_pytorch", methods=["POST"])
def recognize_text_pytorch():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        content = get_prediction(image_bytes=img_bytes)
        return jsonify({"prediction": content})

if __name__ == '__main__':
    app.run(port=5139)

