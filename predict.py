import numpy as np
import tensorflow as tf

from config import CLASS_MAPPING, MODEL_PATH
from preprocessing import preprocess_single_emnist_image


class CharacterPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train the model or copy "
                "best_emnist_model.keras into the models folder."
            )
        self.model = tf.keras.models.load_model(model_path)

    
   def predict_canvas(self, canvas_image):
        """Predict an already-upright 28x28 canvas image."""

        image = np.asarray(canvas_image)

        if image.shape != (28, 28):
            raise ValueError(
                f"Expected (28, 28) image, received {image.shape}"
            )

        # Do NOT transpose: canvas images are already upright.
        batch = image.astype(np.float32) / 255.0
        batch = np.expand_dims(batch, axis=(0, -1))

        scores = self.model.predict(batch, verbose=0)[0]
        top_indices = np.argsort(scores)[-3:][::-1]
        best = int(top_indices[0])

        return {
            "character": CLASS_MAPPING[best],
            "class_index": best,
            "confidence": float(scores[best] * 100),
            "top_3": [
                {
                    "character": CLASS_MAPPING[int(index)],
                    "confidence": float(scores[index] * 100),
                }
                for index in top_indices
            ],
        }
