import argparse
import json
import os
import sys

import cv2
from PIL import Image

import image_detect

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
DEFAULT_FRAMES = 16

_face_cascade = None
_face_cascade_tried = False


def _get_face_cascade():
    global _face_cascade, _face_cascade_tried
    if not _face_cascade_tried:
        _face_cascade_tried = True
        try:
            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                _face_cascade = cascade
        except Exception:
            _face_cascade = None
        if _face_cascade is None:
            print("Note: face detector not available, using whole frames instead.")
    return _face_cascade


def sample_frames(video_path, num_frames=DEFAULT_FRAMES):
    """Pick num_frames frames spread evenly from start to end of the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video: %s" % video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        positions = sorted(set(int(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)))
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    if not frames:
        cap.release()
        cap = cv2.VideoCapture(video_path)
        allf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            allf.append(frame)
        if allf:
            idx = sorted(set(int(i * (len(allf) - 1) / max(num_frames - 1, 1)) for i in range(num_frames)))
            frames = [allf[i] for i in idx]
    cap.release()
    return frames


def crop_face(frame_bgr, margin=0.3):
    """Crop the biggest face (with margin). If no face is found, use the whole frame."""
    h, w = frame_bgr.shape[:2]
    cascade = _get_face_cascade()
    faces = []
    if cascade is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        mx, my = int(fw * margin), int(fh * margin)
        x0, y0 = max(x - mx, 0), max(y - my, 0)
        x1, y1 = min(x + fw + mx, w), min(y + fh + my, h)
        crop = frame_bgr[y0:y1, x0:x1]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), True
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), False


def predict_video(video_path, num_frames=DEFAULT_FRAMES, threshold=image_detect.DEFAULT_THRESHOLD,
                  use_face_crop=True):
    frames = sample_frames(video_path, num_frames)
    if not frames:
        raise ValueError("No frames could be read from the video.")

    images, faces_found = [], 0
    for fr in frames:
        if use_face_crop:
            im, found = crop_face(fr)
            faces_found += int(found)
        else:
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        images.append(im)

    frame_probs = image_detect.score_images(images)
    mean_prob = sum(frame_probs) / len(frame_probs)
    flagged = sum(1 for p in frame_probs if p >= threshold)

    result = image_detect.make_result(mean_prob, threshold)
    result.update({
        "frames_analyzed": len(frame_probs),
        "frames_flagged_fake": flagged,
        "faces_found": faces_found,
        "frame_fake_probabilities": [round(p * 100, 1) for p in frame_probs],
    })
    return result


def _main():
    ap = argparse.ArgumentParser(description="Real vs Fake video detector")
    ap.add_argument("--file", help="path to one video")
    ap.add_argument("--folder", help="path to a folder of videos")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--threshold", type=float, default=image_detect.DEFAULT_THRESHOLD)
    ap.add_argument("--no-face-crop", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.file and not args.folder:
        ap.error("give --file or --folder")

    paths = []
    if args.file:
        paths.append(args.file)
    if args.folder:
        for name in sorted(os.listdir(args.folder)):
            if name.lower().endswith(VIDEO_EXTENSIONS):
                paths.append(os.path.join(args.folder, name))
    if not paths:
        print("No videos found.")
        return 1

    results = []
    for p in paths:
        try:
            r = predict_video(p, args.frames, args.threshold, not args.no_face_crop)
        except Exception as exc:
            print("%s -> ERROR: %s" % (p, exc))
            continue
        r["file"] = p
        results.append(r)
        if not args.json:
            print("%s -> %s (%.1f%% confidence) | %d frames, %d flagged fake, faces found in %d" % (
                os.path.basename(p), r["label"], r["confidence"], r["frames_analyzed"],
                r["frames_flagged_fake"], r["faces_found"]))

    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())