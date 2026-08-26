from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import tempfile
import traceback

from detect import predict


app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Voice Deepfake Detection API is running"
    })


# ---------------------------------------------------------
# DETECT AUDIO FILE
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_audio():

    try:

        # Check whether a file was uploaded
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No audio file was uploaded."
            }), 400

        file = request.files["file"]

        # Check filename
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

        try:

            # -------------------------------------------------
            # Run the AI detector
            # -------------------------------------------------

            label, confidence, prob_dict = predict(temp_path)

            # -------------------------------------------------
            # Safely convert probabilities
            # -------------------------------------------------

            breakdown = {}

            for key, value in prob_dict.items():

                try:

                    # Normal scalar
                    if isinstance(value, (int, float)):
                        numeric_value = float(value)

                    # Handle numpy values
                    elif hasattr(value, "item"):
                        numeric_value = float(value.item())

                    # Handle list / tuple / numpy array
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
            # Normalize label
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
                "label": final_label,
                "confidence": round(confidence, 2),
                "breakdown": breakdown
            }

            print("\n========================================")
            print("VOICE DEEPFAKE DETECTION RESULT")
            print("========================================")
            print(f"Verdict    : {final_label}")
            print(f"Confidence : {confidence:.2f}%")
            print(f"Breakdown  : {breakdown}")
            print("========================================\n")

            return jsonify(result)

        finally:

            # -------------------------------------------------
            # Delete temporary uploaded file
            # -------------------------------------------------

            if os.path.exists(temp_path):

                try:
                    os.remove(temp_path)

                except OSError:
                    pass

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


# ---------------------------------------------------------
# RUN FLASK SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n========================================")
    print("REAL vs FAKE VOICE DETECTOR")
    print("========================================")
    print("Server starting...")
    print("Open: http://127.0.0.1:5000/")
    print("========================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )