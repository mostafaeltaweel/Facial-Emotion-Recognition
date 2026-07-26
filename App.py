import json
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from model import EMOTION_LABELS, load_model
from preprocessing import face_to_tensor

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_model_combined.pth"
TEMPERATURE_PATH = BASE_DIR / "temperature.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.40
DETECT_EVERY_N_FRAMES = 5


@st.cache_resource(show_spinner="Loading the emotion model…")
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing best_model_combined.pth. Put your model file next to app.py "
            "and rename it exactly to best_model_combined.pth."
        )
    return load_model(MODEL_PATH, DEVICE)


@st.cache_resource
def get_detector():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("OpenCV face detector could not be loaded.")
    return cascade


def temperature():
    if not TEMPERATURE_PATH.exists():
        return 1.0
    try:
        return float(json.loads(TEMPERATURE_PATH.read_text(encoding="utf-8")).get("temperature", 1.0))
    except (ValueError, OSError, json.JSONDecodeError):
        return 1.0


def predict(face_bgr):
    image = face_to_tensor(face_bgr).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        return F.softmax(get_model()(image) / temperature(), dim=1)[0].cpu().numpy()


def detect_and_predict(frame):
    detector = get_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    results = []
    for x, y, width, height in faces:
        crop = frame[y:y + height, x:x + width]
        if crop.size == 0:
            continue
        probabilities = predict(crop)
        results.append((x, y, width, height, probabilities))
    return results


def annotate(frame, detections):
    for x, y, width, height, probabilities in detections:
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        is_uncertain = confidence < CONFIDENCE_THRESHOLD
        label = "Uncertain" if is_uncertain else EMOTION_LABELS[index]
        text = f"{label}: {confidence:.0%}"
        color = (0, 165, 255) if is_uncertain else (0, 200, 0)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(frame, text, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_number = 0
        self.detections = []

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self.frame_number += 1
        if self.frame_number % DETECT_EVERY_N_FRAMES == 0 or not self.detections:
            self.detections = detect_and_predict(image)
        annotate(image, self.detections)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.title("🎭 Facial Emotion Recognition")
st.caption("EfficientNet-B3 + CBAM — live webcam or image upload")

if not MODEL_PATH.exists():
    st.error("لم يتم العثور على الموديل. أعد تسمية الملف إلى best_model_combined.pth وضعه بجانب app.py.")
    st.stop()

mode = st.radio("اختر الإدخال", ["كاميرا مباشرة", "رفع صورة"], horizontal=True)

if mode == "كاميرا مباشرة":
    st.info("اضغط START ثم اسمح للمتصفح باستخدام الكاميرا. إن لم تظهر، افتح الرابط في Chrome أو Edge وتحقق من إذن الكاميرا.")
    webrtc_streamer(
        key="emotion-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
else:
    upload = st.file_uploader("ارفع صورة بها وجه", type=["jpg", "jpeg", "png"])
    if upload:
        raw = np.frombuffer(upload.read(), np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            st.error("تعذر قراءة الصورة.")
        else:
            result = annotate(image, detect_and_predict(image))
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
