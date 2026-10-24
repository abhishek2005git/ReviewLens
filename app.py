"""Run with: python app.py"""

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from model import load_model, predict_review


def create_app(artifact=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    artifact = load_model() if artifact is None else artifact

    @app.get("/")
    def index():
        return render_template("index.html", mode=artifact["mode"],
                               model_name=artifact.get("model_name", "TF-IDF + Logistic Regression"),
                               training_rows=f'{artifact["training_rows"]:,}')

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", mode=artifact["mode"],
                       training_rows=artifact["training_rows"],
                       model_name=artifact.get("model_name", "TF-IDF + Logistic Regression"),
                       metrics=artifact.get("metrics"))

    @app.post("/api/predict")
    def predict():
        if not request.is_json:
            return jsonify(error="Send a JSON object with a review field."), 415
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify(error="Send a JSON object with a review field."), 400
        try:
            return jsonify(predict_review(artifact, data.get("review")))
        except ValueError as error:
            return jsonify(error=str(error)), 400

    @app.errorhandler(BadRequest)
    def bad_json(error):
        return jsonify(error="The request contains invalid JSON."), 400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="This request is too large. Reviews can contain up to 5,000 characters."), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5001, debug=False)
