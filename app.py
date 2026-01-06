import base64

from flask import Flask, jsonify, render_template, request
import cv2
import numpy as np

import angle_finder

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/measure", methods=["POST"])
def measure():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        result = angle_finder.analyze_image_array(image)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    annotated = result.pop("annotated")

    success, buffer = cv2.imencode(".png", annotated)
    if not success:
        return jsonify({"error": "Failed to encode annotated image"}), 500

    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    data_url = "data:image/png;base64," + encoded

    return jsonify(
        {
            "center": result["center"],
            "radius": result["radius"],
            "angles": result["angles"],
            "sector_angles": result["sector_angles"],
            "annotated_image": data_url,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
