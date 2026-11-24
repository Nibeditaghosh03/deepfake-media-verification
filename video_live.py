import argparse
import time
import threading
from collections import deque

import cv2
from PIL import Image

import image_detect


def main():
    ap = argparse.ArgumentParser(description="Live webcam real vs fake detector")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between predictions")
    ap.add_argument("--window", type=int, default=5, help="predictions averaged for the label")
    ap.add_argument("--threshold", type=float, default=image_detect.DEFAULT_THRESHOLD)
    args = ap.parse_args()

    print("Loading model...")
    image_detect._load()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: could not open camera %d. Close other apps using it, or try --camera 1" % args.camera)
        return 1

    face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    scores = deque(maxlen=args.window)
    state = {"busy": False}

    def worker(img):
        try:
            scores.append(image_detect.score_images([img])[0])
        except Exception as exc:
            print("prediction error:", exc)
        finally:
            state["busy"] = False

    last = 0.0
    print("Camera started. Click the video window and press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: could not read from camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_det.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        box = None
        if len(faces) > 0:
            box = max(faces, key=lambda b: b[2] * b[3])

        if box is None:
            text, color = "No face found", (0, 200, 255)
        else:
            x, y, w, h = [int(v) for v in box]
            now = time.time()
            if not state["busy"] and now - last >= args.interval:
                pad = int(0.2 * w)
                crop = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                state["busy"] = True
                last = now
                threading.Thread(target=worker, args=(img,), daemon=True).start()

            if scores:
                r = image_detect.make_result(sum(scores) / len(scores), args.threshold)
                text = "%s %.1f%%" % (r["label"], r["confidence"])
                color = (0, 0, 255) if r["label"] == "FAKE" else (0, 180, 0)
            else:
                text, color = "Analysing...", (200, 200, 200)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Live Deepfake Detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())