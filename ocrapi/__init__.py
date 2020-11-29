import os
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request
from werkzeug.utils import secure_filename

from .lp_prediction_pytorch.lp_prediction import get_prediction

# from .pl_detection.pl_detection import pl_detection

# import io
# from PIL import Image
# import numpy as np

UPLOAD_FOLDER = Path(__file__).resolve().parent / "static" / "uploads"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "static" / "outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY="dev",
        DATABASE=os.path.join(app.instance_path, "ocrapi.sqlite"),
        UPLOAD_FOLDER=UPLOAD_FOLDER,
        OUTPUT_FOLDER=OUTPUT_FOLDER,
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says welcome
    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/welcome")
    def welcome():
        return "Welcome dear developers!"

    @app.route("/gimme_your_plate", methods=["GET", "POST"])
    def plate():
        if request.method == "POST":
            # check if the post request has the file part
            if "file" not in request.files:
                flash("No file part")
                return redirect(request.url)
            file = request.files["file"]
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == "":
                flash("No selected file")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                # load uploaded file
                filename = secure_filename(file.filename)
                original_path = app.config["UPLOAD_FOLDER"] / filename
                original_rel_path = Path("..") / "static" / "uploads" / filename
                file.save(original_path)
                # FIXME: plate detection
                # file = open(original_path, "rb")
                # img_bytes = file.read()
                # img = np.array(Image.open(io.BytesIO(img_bytes)))
                # img = img[:, np.newaxis]
                filename = "extracted_plate.png"
                # plate_path = app.config["OUTPUT_FOLDER"] / filename
                plate_rel_path = Path("..") / "static" / "outputs" / filename
                # plate_file = pl_detection(img)
                # plate_file.save(plate_path)
                # content extraction
                # input_img = open(plate_path, "rb")
                # img_bytes = input_img.read()
                # response = get_prediction(image_bytes=img_bytes)
                response = (
                    None  # TODO: remove this line when everything is working
                )
                return render_template(
                    "license_plate.html",
                    original_img=original_rel_path,
                    plate_img=plate_rel_path,
                    response=response,
                )
        return render_template("license_plate.html")

    @app.route("/gimme_your_letter")
    def letter():
        return render_template("handwritten.html")

    @app.route("/recognize_text_pytorch", methods=["POST"])
    def recognize_text_pytorch():
        if request.method == "POST":
            file = request.files["file"]
            img_bytes = file.read()
            content = get_prediction(image_bytes=img_bytes)
            return jsonify({"prediction": content})

    return app
