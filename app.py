import base64
import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

from predict import CharacterPredictor

app = Flask(__name__)
predictor = CharacterPredictor()


def preprocess_canvas_image(image_bytes):
    """
    Initial web-canvas adapter.

    Note: this is NOT identical to EMNIST's raw-image preprocessing.
    It crops, scales and centers a browser drawing. Validate and tune this
    mapping using real drawings before claiming EMNIST-level accuracy.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Browser canvas: black background with white strokes.
    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)

    image = image.point(lambda pixel: 255 if pixel > 25 else 0)
    bounds = image.getbbox()
    if bounds is None:
        raise ValueError("The canvas is empty. Draw a character first.")

    image = image.crop(bounds)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image", "")

    if not data_url.startswith("data:image/") or "," not in data_url:
        return jsonify({"error": "A valid canvas image is required."}), 400

    try:
        _, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded, validate=True)
        raw_image = preprocess_canvas_image(image_bytes)
        result = predictor.predict_canvas(raw_image)
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the server log."}), 500

    




# @app.route("/predict", methods=["POST"])
# def predict():
#     app.logger.info("STEP 1: Prediction request received")

#     payload = request.get_json(silent=True) or {}
#     data_url = payload.get("image", "")

#     if not data_url.startswith("data:image/") or "," not in data_url:
#         app.logger.warning("Invalid image payload")
#         return jsonify({"error": "A valid canvas image is required."}), 400

#     try:
#         _, encoded = data_url.split(",", 1)
#         image_bytes = base64.b64decode(encoded, validate=True)

#         app.logger.info("STEP 2: Image decoded")

#         raw_image = preprocess_canvas_image(image_bytes)

#         app.logger.info("STEP 3: Image preprocessed")
#         app.logger.info("STEP 4: Starting TensorFlow prediction")

#         result = predictor.predict(raw_image)

#         app.logger.info("STEP 5: Prediction completed")

#         return jsonify(result)

#     except ValueError as error:
#         app.logger.exception("Image processing error")
#         return jsonify({"error": str(error)}), 400

#     except Exception:
#         app.logger.exception("Prediction failed")
#         return jsonify({"error": "Prediction failed. Check the server log."}), 500



if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=False)
