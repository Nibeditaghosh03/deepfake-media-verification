try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return 0
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import os
import tempfile
import traceback

from detect import predict
from text_detect import detect_text


app = Flask(__name__)
CORS(app)

from live_routes import live_bp
app.register_blueprint(live_bp)

from media_routes import media_bp
app.register_blueprint(media_bp)    


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Voice and Text Deepfake Detection API is running"
    })


# =========================================================
# VOICE DEEPFAKE DETECTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict_audio():

    temp_path = None

    try:

        # -------------------------------------------------
        # Check whether a file was uploaded
        # -------------------------------------------------

        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No audio file was uploaded."
            }), 400

        file = request.files["file"]

        # -------------------------------------------------
        # Check filename
        # -------------------------------------------------

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        # -------------------------------------------------
        # Save uploaded file temporarily
        # -------------------------------------------------

        suffix = os.path.splitext(file.filename)[1]

        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            file.save(temp_file.name)
            temp_path = temp_file.name

        # -------------------------------------------------
        # Run voice AI detector
        # -------------------------------------------------

        label, confidence, prob_dict = predict(temp_path)

        # -------------------------------------------------
        # Safely convert probabilities
        # -------------------------------------------------

        breakdown = {}

        if prob_dict:

            for key, value in prob_dict.items():

                try:

                    # Normal number
                    if isinstance(value, (int, float)):
                        numeric_value = float(value)

                    # NumPy scalar
                    elif hasattr(value, "item"):
                        numeric_value = float(value.item())

                    # List / tuple / array
                    elif hasattr(value, "__len__"):

                        if len(value) > 0:

                            first_value = value[0]

                            if hasattr(first_value, "item"):
                                numeric_value = float(first_value.item())
                            else:
                                numeric_value = float(first_value)

                        else:
                            numeric_value = 0.0

                    else:
                        numeric_value = float(value)

                    breakdown[str(key)] = round(
                        numeric_value * 100,
                        2
                    )

                except (TypeError, ValueError, IndexError):

                    breakdown[str(key)] = str(value)

        # -------------------------------------------------
        # Normalize voice label
        # -------------------------------------------------

        label = str(label).upper()

        if "FAKE" in label or "SPOOF" in label:
            final_label = "FAKE"

        elif "REAL" in label or "BONA" in label:
            final_label = "REAL"

        else:
            final_label = label

        # -------------------------------------------------
        # Normalize confidence
        # -------------------------------------------------

        try:
            confidence = float(confidence)

        except (TypeError, ValueError):
            confidence = 0.0

        # -------------------------------------------------
        # Final response
        # -------------------------------------------------

        result = {
            "success": True,
            "type": "voice",
            "label": final_label,
            "confidence": round(confidence, 2),
            "breakdown": breakdown
        }

        # -------------------------------------------------
        # Terminal output
        # -------------------------------------------------

        print("\n========================================")
        print("VOICE DEEPFAKE DETECTION RESULT")
        print("========================================")
        print(f"Verdict    : {final_label}")
        print(f"Confidence : {confidence:.2f}%")
        print(f"Breakdown  : {breakdown}")
        print("========================================\n")

        return jsonify(result)

    except Exception as error:

        print("\n========================================")
        print("ERROR IN /predict")
        print("========================================")

        traceback.print_exc()

        print("========================================\n")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

    finally:

        # -------------------------------------------------
        # Delete temporary uploaded file
        # -------------------------------------------------

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except OSError:
                pass


# =========================================================
# TEXT DEEPFAKE DETECTION
# =========================================================

@app.route("/predict-text", methods=["POST"])
def predict_text():

    try:

        # -------------------------------------------------
        # Get JSON data
        # -------------------------------------------------

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data was provided."
            }), 400

        # -------------------------------------------------
        # Get text
        # -------------------------------------------------

        text = data.get("text", "")

        if not text or not text.strip():

            return jsonify({
                "success": False,
                "error": "No text was provided."
            }), 400

        text = text.strip()

        # -------------------------------------------------
        # Run text AI detector
        # -------------------------------------------------

        result = detect_text(text)

        # -------------------------------------------------
        # Make sure result exists
        # -------------------------------------------------

        if result is None:

            return jsonify({
                "success": False,
                "error": "Text detector did not return a result."
            }), 500

        # -------------------------------------------------
        # Extract result
        # -------------------------------------------------

        label = result.get("label", "UNKNOWN")
        confidence = result.get("confidence", 0.0)

        # -------------------------------------------------
        # Normalize label
        # -------------------------------------------------

        label_upper = str(label).upper()

        if (
            "AI" in label_upper
            or "FAKE" in label_upper
            or "MACHINE" in label_upper
            or "GENERATED" in label_upper
        ):

            final_label = "AI-GENERATED"

        elif (
            "HUMAN" in label_upper
            or "REAL" in label_upper
        ):

            final_label = "HUMAN"

        else:

            final_label = label_upper

        # -------------------------------------------------
        # Normalize confidence
        # -------------------------------------------------

        try:
            confidence = float(confidence)

        except (TypeError, ValueError):
            confidence = 0.0

        # -------------------------------------------------
        # Final response
        # -------------------------------------------------

        response = {
            "success": True,
            "type": "text",
            "label": final_label,
            "confidence": round(confidence, 2),
            "text": text
        }

        # -------------------------------------------------
        # Terminal output
        # -------------------------------------------------

        print("\n========================================")
        print("TEXT DEEPFAKE DETECTION RESULT")
        print("========================================")
        print(f"Text       : {text}")
        print(f"Verdict    : {final_label}")
        print(f"Confidence : {confidence:.2f}%")
        print("========================================\n")

        return jsonify(response)

    except Exception as error:

        print("\n========================================")
        print("ERROR IN /predict-text")
        print("========================================")

        traceback.print_exc()

        print("========================================\n")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# =========================================================
# RUN FLASK SERVER
# =========================================================

if __name__ == "__main__":

    print("\n========================================")
    print("REAL vs FAKE MEDIA VERIFICATION")
    print("========================================")
    print("Server starting...")
    print("Open: http://127.0.0.1:5000/")
    print("========================================\n")

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )