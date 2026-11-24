import os
import tempfile

from flask import Blueprint, jsonify, request
from PIL import Image

import image_detect
import video_detect

media_bp = Blueprint("media", __name__)


def _threshold():
    try:
        return float(request.form.get("threshold", image_detect.DEFAULT_THRESHOLD))
    except ValueError:
        return image_detect.DEFAULT_THRESHOLD


@media_bp.route("/api/image", methods=["POST"])
def api_image():
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"error": "No image uploaded."}), 400
    if not f.filename.lower().endswith(image_detect.IMAGE_EXTENSIONS):
        return jsonify({"error": "Please upload a JPG, PNG, BMP or WEBP image."}), 400
    try:
        img = Image.open(f.stream)
        img.load()
        return jsonify(image_detect.predict_image(img, _threshold()))
    except Exception as exc:
        return jsonify({"error": "Could not analyse the image: %s" % exc}), 500


@media_bp.route("/api/video", methods=["POST"])
def api_video():
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"error": "No video uploaded."}), 400
    if not f.filename.lower().endswith(video_detect.VIDEO_EXTENSIONS):
        return jsonify({"error": "Please upload an MP4, AVI, MOV, MKV or WEBM video."}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        f.save(tmp_path)
        return jsonify(video_detect.predict_video(tmp_path, threshold=_threshold()))
    except Exception as exc:
        return jsonify({"error": "Could not analyse the video: %s" % exc}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Image and Video Detection</title>
<style>
 body{font-family:Arial,sans-serif;max-width:720px;margin:30px auto;padding:0 16px;color:#222}
 .tabs button{padding:10px 22px;border:1px solid #888;background:#eee;cursor:pointer;font-size:15px}
 .tabs button.active{background:#2b5fd9;color:#fff;border-color:#2b5fd9}
 .panel{display:none;border:1px solid #ccc;padding:20px;margin-top:-1px}
 .panel.active{display:block}
 button.go{margin-top:14px;padding:9px 20px;font-size:15px;cursor:pointer}
 #imgPrev,#vidPrev{max-width:100%;max-height:300px;margin-top:12px;display:none}
 .result{margin-top:16px;padding:14px;border-radius:6px;display:none;font-size:15px}
 .fake{background:#fde8e8;border:1px solid #d33}.real{background:#e6f6e6;border:1px solid #2a8a2a}
 .err{background:#fff3cd;border:1px solid #c9a100}
 .big{font-size:24px;font-weight:bold}
</style></head><body>
<h2>Image and Video Detection</h2>
<div class="tabs">
  <button id="tabImg" class="active" onclick="show('img')">Image</button>
  <button id="tabVid" onclick="show('vid')">Video</button>
</div>

<div id="panelImg" class="panel active">
  <p>Upload an image (JPG / PNG). Works best with faces.</p>
  <input type="file" id="imgFile" accept="image/*" onchange="preview('img')"><br>
  <img id="imgPrev">
  <br><button class="go" onclick="run('img')">Check image</button>
  <div id="imgOut" class="result"></div>
</div>

<div id="panelVid" class="panel">
  <p>Upload a short video (MP4 / MOV / AVI). Frames are sampled and checked one by one.</p>
  <input type="file" id="vidFile" accept="video/*" onchange="preview('vid')"><br>
  <video id="vidPrev" controls></video>
  <br><button class="go" onclick="run('vid')">Check video</button>
  <div id="vidOut" class="result"></div>
</div>

<script>
function show(w){
  for (const k of ['img','vid']){
    document.getElementById('panel'+(k=='img'?'Img':'Vid')).classList.toggle('active', k==w);
    document.getElementById('tab'+(k=='img'?'Img':'Vid')).classList.toggle('active', k==w);
  }
}
function preview(k){
  const f = document.getElementById(k+'File').files[0];
  const el = document.getElementById(k+'Prev');
  if (f){ el.src = URL.createObjectURL(f); el.style.display='block'; }
}
async function run(k){
  const f = document.getElementById(k+'File').files[0];
  const out = document.getElementById(k+'Out');
  out.style.display='block'; out.className='result err';
  if(!f){ out.textContent='Please choose a file first.'; return; }
  out.textContent = 'Analysing... the first run can take a minute while the model loads.';
  const fd = new FormData(); fd.append('file', f);
  try{
    const r = await fetch('/api/'+(k=='img'?'image':'video'), {method:'POST', body: fd});
    const d = await r.json();
    if(d.error){ out.textContent = d.error; return; }
    out.className = 'result ' + (d.label=='FAKE' ? 'fake' : 'real');
    let html = '<div class="big">' + d.label + '</div>' +
      'Confidence: ' + d.confidence + '%<br>' +
      'Fake probability: ' + d.fake_probability + '% | Real probability: ' + d.real_probability + '%';
    if(k=='vid'){
      html += '<br>Frames analysed: ' + d.frames_analyzed + ' | flagged fake: ' + d.frames_flagged_fake +
              ' | faces found: ' + d.faces_found;
    }
    out.innerHTML = html;
  }catch(e){ out.textContent = 'Request failed: ' + e; }
}
</script></body></html>"""


@media_bp.route("/media")
def media_page():
    return PAGE