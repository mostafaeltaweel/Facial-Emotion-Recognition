"""
Facial Emotion Recognition - Streamlit App
Graduation Project

Pipeline:
  1. Detect face(s) using OpenCV Haar Cascade
  2. Crop the face region
  3. Apply the SAME preprocessing used in training (bilateral filter + CLAHE + gamma)
  4. Resize to 224x224 and normalize
  5. Run through EfficientNet-B3 + CBAM model
  6. Draw bounding box + emotion label + confidence on the frame
  7. If confidence is too low -> show "Uncertain" instead of forcing a wrong label
"""

import json
import os
import time
from collections import deque

import av
import cv2
import numpy as np
import requests
import streamlit as st
import torch
import torch.nn.functional as F
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

from model import load_model, EMOTION_LABELS, EMOTION_EMOJI
from preprocessing import face_crop_to_tensor

# ─────────────────────────── Configuration ───────────────────────────
MODEL_PATH = "best_model_combined.pth"  # local filename the model is saved/downloaded as
TEMPERATURE_PATH = "temperature.json"   # produced by the training notebook (calibration)

# ↓↓↓ PASTE YOUR GITHUB RELEASE LINKS HERE (leave "" to disable auto-download) ↓↓↓
MODEL_URL = "https://github.com/USERNAME/REPO/releases/download/v1.0/best_model_combined.pth"
TEMPERATURE_URL = "https://github.com/USERNAME/REPO/releases/download/v1.0/temperature.json"
# ↑↑↑ replace USERNAME/REPO/v1.0 with your actual repo + release tag ↑↑↑

CONFIDENCE_THRESHOLD = 0.40             # below this -> "Uncertain"
DETECT_EVERY_N_FRAMES = 5               # throttling: run the model every N frames
SMOOTHING_WINDOW = 6                    # live camera: average probs over the last N detections per face
BOX_COLOR = (0, 200, 0)                 # BGR green
UNCERTAIN_COLOR = (0, 165, 255)         # BGR orange

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_haarcascade_path():
    """Locates haarcascade_frontalface_default.xml. Some opencv-python-headless
    builds on Streamlit Cloud don't expose cv2.data (AttributeError), so we
    fall back to searching the cv2 install directory directly."""
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(path):
            return path
    except AttributeError:
        pass

    # Fallback: look inside the installed cv2 package folder manually
    cv2_dir = os.path.dirname(cv2.__file__)
    for candidate in [
        os.path.join(cv2_dir, "data", "haarcascade_frontalface_default.xml"),
        os.path.join(cv2_dir, "..", "cv2", "data", "haarcascade_frontalface_default.xml"),
    ]:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate haarcascade_frontalface_default.xml in the opencv "
        "installation. Try pinning 'opencv-python-headless==4.9.0.80' in requirements.txt."
    )


FACE_CASCADE = cv2.CascadeClassifier(get_haarcascade_path())


def download_if_missing(local_path, url, label):
    """Downloads a file from a direct URL (e.g. a GitHub Release asset) if it
    isn't already sitting next to app.py. Shows a progress bar since the
    model file can be tens of MBs. Safe to call every run — it's a no-op
    once the file exists locally."""
    if os.path.exists(local_path) or not url:
        return os.path.exists(local_path)

    try:
        with st.spinner(f"جاري تحميل {label} أول مرة فقط (لن يتكرر لاحقًا)..."):
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            progress = st.progress(0)
            downloaded = 0
            tmp_path = local_path + ".part"
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress.progress(min(downloaded / total, 1.0))
            os.rename(tmp_path, local_path)
            progress.empty()
        return True
    except Exception as e:
        st.error(f"تعذّر تحميل {label} تلقائيًا: {e}\n"
                  f"حمّله يدويًا وحطه بجانب app.py باسم '{local_path}'.")
        return False


def load_temperature():
    """Loads the calibration temperature saved by the training notebook.
    Falls back to 1.0 (no calibration) if the file isn't there yet."""
    download_if_missing(TEMPERATURE_PATH, TEMPERATURE_URL, "ملف المعايرة (temperature.json)")
    if os.path.exists(TEMPERATURE_PATH):
        with open(TEMPERATURE_PATH) as f:
            return float(json.load(f).get("temperature", 1.0))
    return 1.0


TEMPERATURE = load_temperature()

# ─────────────────────────── Model loading (cached) ───────────────────────────
@st.cache_resource
def get_model():
    if not download_if_missing(MODEL_PATH, MODEL_URL, "الموديل (best_model_combined.pth)"):
        st.stop()  # no local file AND download failed -> nothing we can do
    return load_model(MODEL_PATH, device=DEVICE)


def predict_emotion(model, face_bgr, use_tta=False):
    """Runs the model on a cropped BGR face image. Returns (probs: np.array[7]).

    use_tta=True averages the prediction on the image AND its horizontal mirror
    (Test-Time Augmentation) — a few extra ms per image for a small, free accuracy
    boost. Used for the (single-shot) image upload mode; skipped in the live camera
    loop where per-frame smoothing already provides a similar stabilizing effect.

    TEMPERATURE (learned during training via Temperature Scaling) is applied so the
    confidence values are calibrated — makes CONFIDENCE_THRESHOLD mean what it says.
    """
    tensor = face_crop_to_tensor(face_bgr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        if use_tta:
            flipped = torch.flip(tensor, dims=[3])
            logits_flip = model(flipped)
            probs = (F.softmax(logits / TEMPERATURE, dim=1) +
                     F.softmax(logits_flip / TEMPERATURE, dim=1)) / 2
        else:
            probs = F.softmax(logits / TEMPERATURE, dim=1)
        probs = probs[0].cpu().numpy()
    return probs


def draw_result(frame_bgr, x, y, w, h, probs):
    """Draws bounding box + top label + confidence on the frame (in-place)."""
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])

    if top_conf < CONFIDENCE_THRESHOLD:
        label = f"Uncertain ({top_conf*100:.0f}%)"
        color = UNCERTAIN_COLOR
    else:
        label = f"{EMOTION_LABELS[top_idx]} ({top_conf*100:.0f}%)"
        color = BOX_COLOR

    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr, (x, y - text_h - 12), (x + text_w + 6, y), color, -1)
    cv2.putText(frame_bgr, label, (x + 3, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame_bgr


class FaceTracker:
    """Very lightweight tracker: matches each new detection to the closest one
    from previous detections (by box-center distance) and keeps a rolling
    average of its probability vector. This is what removes the "flicker"
    where the live label jumps between e.g. Happy/Neutral every detection —
    a much bigger perceived-quality win than TTA for a live video feed."""

    def __init__(self, window=SMOOTHING_WINDOW, max_distance=80):
        self.window = window
        self.max_distance = max_distance
        self.tracks = {}  # id -> {'center': (cx, cy), 'probs': deque}
        self.next_id = 0

    def update(self, detections):
        """detections: list of (x, y, w, h, probs). Returns the same list but
        with each probs vector replaced by its smoothed rolling average."""
        used_ids = set()
        smoothed_detections = []
        for (x, y, w, h, probs) in detections:
            cx, cy = x + w / 2, y + h / 2
            best_id, best_dist = None, self.max_distance
            for tid, track in self.tracks.items():
                if tid in used_ids:
                    continue
                dist = ((track['center'][0] - cx) ** 2 + (track['center'][1] - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {'center': (cx, cy), 'probs': deque(maxlen=self.window)}

            self.tracks[best_id]['center'] = (cx, cy)
            self.tracks[best_id]['probs'].append(probs)
            used_ids.add(best_id)

            smoothed_probs = np.mean(self.tracks[best_id]['probs'], axis=0)
            smoothed_detections.append((x, y, w, h, smoothed_probs))

        # Drop tracks that weren't matched this round (face left the frame)
        self.tracks = {tid: t for tid, t in self.tracks.items() if tid in used_ids}
        return smoothed_detections


# ─────────────────────────── Live webcam processor ───────────────────────────
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = get_model()
        self.frame_count = 0
        self.tracker = FaceTracker()
        self.cached_faces = []  # list of (x, y, w, h, probs) reused between detections

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1
        run_inference = (self.frame_count % DETECT_EVERY_N_FRAMES == 0)

        if run_inference:
            faces = FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            raw_detections = []
            for (x, y, w, h) in faces:
                face_crop = img[y:y + h, x:x + w]
                if face_crop.size == 0:
                    continue
                probs = predict_emotion(self.model, face_crop)
                raw_detections.append((x, y, w, h, probs))
            # Smooth over the last few detections per tracked face -> stable label
            self.cached_faces = self.tracker.update(raw_detections)

        # Draw using the latest (smoothed) cached results on every frame
        for (x, y, w, h, probs) in self.cached_faces:
            draw_result(img, x, y, w, h, probs)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────── Streamlit UI ───────────────────────────
st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭", layout="centered")
st.title("🎭 Facial Emotion Recognition")
st.caption("EfficientNet-B3 + CBAM | Graduation Project")

mode = st.radio("اختر طريقة الإدخال:", ["📷 كاميرا مباشرة (Live)", "🖼️ رفع صورة"], horizontal=True)

st.divider()

if mode == "📷 كاميرا مباشرة (Live)":
    st.info("اضغط START وامنح المتصفح إذن الوصول للكاميرا.")
    webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

else:
    uploaded_file = st.file_uploader("ارفع صورة تحتوي على وجه", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        model = get_model()
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            st.warning("لم يتم العثور على أي وجه في الصورة. جرّب صورة أوضح.")
        else:
            last_probs = None
            for (x, y, w, h) in faces:
                face_crop = img_bgr[y:y + h, x:x + w]
                # A single uploaded image can afford the extra TTA inference pass
                probs = predict_emotion(model, face_crop, use_tta=True)
                draw_result(img_bgr, x, y, w, h, probs)
                last_probs = probs  # for single-face display below

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="النتيجة", use_container_width=True)

            if last_probs is not None:
                st.subheader("احتمالية كل مشاعر (لآخر وجه تم اكتشافه):")
                for label, p in sorted(zip(EMOTION_LABELS, last_probs), key=lambda t: -t[1]):
                    st.write(f"{EMOTION_EMOJI[label]} **{label}**")
                    st.progress(float(p))
