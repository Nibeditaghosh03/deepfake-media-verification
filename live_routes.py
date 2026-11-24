import base64

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request
from PIL import Image

import image_detect

live_bp = Blueprint("live", __name__)

_face_det = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _find_face(bgr):
    """Return (PIL face crop, (x, y, w, h)) or (None, None)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_det.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None, None
    x, y, w, h = [int(v) for v in max(faces, key=lambda b: b[2] * b[3])]
    pad = int(0.3 * w)
    H, W = bgr.shape[:2]
    crop = bgr[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return img, (x, y, w, h)


@live_bp.route("/live")
def live_page():
    return render_template("live.html")


@live_bp.route("/live-predict", methods=["POST"])
def live_predict():
    data = request.get_json(silent=True) or {}
    try:
        raw = base64.b64decode(data["image"].split(",", 1)[1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode frame")
    except Exception as exc:
        return jsonify({"error": "bad image: %s" % exc}), 400

    img, box = _find_face(frame)
    if img is None:
        return jsonify({"face": False})

    fake_prob = image_detect.score_images([img])[0]
    result = image_detect.make_result(fake_prob)
    result.update(face=True, box=list(box))
    return jsonify(result)