import os

from flask import Flask, jsonify, request

from .lp_prediction_pytorch.lp_prediction import get_prediction


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY="dev",
        DATABASE=os.path.join(app.instance_path, "ocrapi.sqlite"),
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
    @app.route("/welcome")
    def home():
        return "Welcome dear developers!"

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "POST":
            file = request.files["file"]
            img_bytes = file.read()
            content = get_prediction(image_bytes=img_bytes)
            return jsonify({"prediction": content})
        else:
            return "Nothing to say"

    return app
