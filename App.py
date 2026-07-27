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

NOTE: Live continuous video (streamlit-webrtc) was tried but is unreliable on
Streamlit Community Cloud's free tier (a long-standing, documented platform
limitation, not a bug in this code). Instead, this app uses st.camera_input:
the user takes a snapshot, gets an instant analysis, and can retake it in one
click for a "semi-live" experience that works reliably everywhere.
"""

import io
import json
import os
import tempfile

import cv2
import numpy as np
import requests
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from model import load_model, EMOTION_LABELS, EMOTION_EMOJI
from preprocessing import face_crop_to_tensor

# ─────────────────────────── Configuration ───────────────────────────
MODEL_PATH = "best_model_combined.pth"  # local filename the model is saved/downloaded as
TEMPERATURE_PATH = "temperature.json"   # produced by the training notebook (calibration)

# ↓↓↓ Direct download link for the trained model (GitHub Release asset) ↓↓↓
MODEL_URL = "https://github.com/mostafaeltaweel/Facial-Emotion-Recognition/releases/download/v1.0/best_model_combined.pth"
# No temperature.json was uploaded to the release, so this stays empty —
# the app will just use TEMPERATURE = 1.0 (no calibration) until you add one.
TEMPERATURE_URL = ""

CONFIDENCE_THRESHOLD = 0.40   # below this -> "Uncertain"
BOX_COLOR = (0, 200, 0)       # BGR green
UNCERTAIN_COLOR = (0, 165, 255)  # BGR orange

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

    cv2_dir = os.path.dirname(cv2.__file__)
    for candidate in [
        os.path.join(cv2_dir, "data", "haarcascade_frontalface_default.xml"),
        os.path.join(cv2_dir, "..", "cv2", "data", "haarcascade_frontalface_default.xml"),
    ]:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate haarcascade_frontalface_default.xml in the opencv installation."
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
        st.stop()
    return load_model(MODEL_PATH, device=DEVICE)


def predict_emotion(model, face_bgr, use_tta=True):
    """Runs the model on a cropped BGR face image. Returns probs: np.array[7].

    use_tta=True averages the prediction on the image AND its horizontal mirror
    (Test-Time Augmentation) — a small, free accuracy boost. Cheap enough to
    always use here since we only run it once per snapshot (not per video frame).

    TEMPERATURE (learned during training via Temperature Scaling) is applied so
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


def analyze_and_display(model, img_bgr):
    """Shared pipeline for both camera snapshots and uploaded images:
    detect faces -> predict -> draw boxes -> show result image + probability bars."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        st.warning("لم يتم العثور على أي وجه في الصورة. جرّب صورة أوضح أو أقرب للكاميرا.")
        return

    last_probs = None
    for (x, y, w, h) in faces:
        face_crop = img_bgr[y:y + h, x:x + w]
        probs = predict_emotion(model, face_crop, use_tta=True)
        draw_result(img_bgr, x, y, w, h, probs)
        last_probs = probs

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="النتيجة", use_container_width=True)

    if last_probs is not None:
        st.subheader("احتمالية كل مشاعر (لآخر وجه تم اكتشافه):")
        for label, p in sorted(zip(EMOTION_LABELS, last_probs), key=lambda t: -t[1]):
            st.write(f"{EMOTION_EMOJI[label]} **{label}**")
            st.progress(float(p))


def analyze_video_dense(model, video_path, target_fps=8, max_frames=200):
    """For SHORT clips with fast-changing emotions: samples frames much more
    densely (several per second instead of one per minute), draws the box on
    every sampled frame, and logs only the moments where the detected emotion
    actually CHANGES (not every single frame — avoids a noisy repeated log).

    Returns: (annotated_frames_rgb: list[np.ndarray], change_log: list[(timestamp_str, label, confidence)])
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("تعذّر فتح ملف الفيديو. جرّب صيغة mp4.")
        return [], []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / target_fps))

    annotated_frames = []
    change_log = []
    last_label = None

    progress = st.progress(0.0)
    frame_idx = 0
    processed = 0
    while processed < max_frames and frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        seconds = frame_idx / src_fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        label, confidence = "لا يوجد وجه", 0.0
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = frame[y:y + h, x:x + w]
            probs = predict_emotion(model, face_crop, use_tta=False)
            draw_result(frame, x, y, w, h, probs)
            top_idx = int(np.argmax(probs))
            label = EMOTION_LABELS[top_idx]
            confidence = float(probs[top_idx])

        annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if label != last_label:
            mins, secs = divmod(seconds, 60)
            timestamp_str = f"{int(mins):02d}:{secs:05.2f}"
            change_log.append((timestamp_str, label, confidence))
            last_label = label

        processed += 1
        frame_idx += step
        progress.progress(min(processed / min(max_frames, total_frames // step + 1), 1.0))

    cap.release()
    progress.empty()
    return annotated_frames, change_log


def frames_to_gif_bytes(frames_rgb, fps):
    """Builds an in-memory animated GIF from RGB frames. GIF is used instead of
    an encoded video file because browsers reliably play back GIFs everywhere,
    while OpenCV's mp4 writer (mp4v codec) often fails to play in-browser."""
    pil_frames = [Image.fromarray(f) for f in frames_rgb]
    buffer = io.BytesIO()
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        buffer, format="GIF", save_all=True,
        append_images=pil_frames[1:], duration=duration_ms, loop=0,
    )
    return buffer.getvalue()



    """Samples frames from a video file every `interval_seconds` seconds,
    runs face detection + emotion prediction on each sample, and returns a
    timeline: list of (timestamp_str, label, confidence, thumbnail_rgb)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("تعذّر فتح ملف الفيديو. جرّب صيغة mp4.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps * interval_seconds))
    total_samples = max(1, total_frames // frame_step)

    timeline = []
    progress = st.progress(0.0)
    status = st.empty()

    frame_idx = 0
    sample_count = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        seconds = frame_idx / fps
        mins, secs = divmod(int(seconds), 60)
        timestamp_str = f"{mins:02d}:{secs:02d}"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
            face_crop = frame[y:y + h, x:x + w]
            probs = predict_emotion(model, face_crop, use_tta=False)
            draw_result(frame, x, y, w, h, probs)
            top_idx = int(np.argmax(probs))
            label = EMOTION_LABELS[top_idx]
            confidence = float(probs[top_idx])
        else:
            label, confidence = "لا يوجد وجه", 0.0

        thumb_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timeline.append((timestamp_str, label, confidence, thumb_rgb))

        sample_count += 1
        status.text(f"جاري التحليل... {timestamp_str} ({sample_count}/{total_samples})")
        progress.progress(min(sample_count / total_samples, 1.0))

        frame_idx += frame_step
        if frame_idx >= total_frames:
            break

    cap.release()
    progress.empty()
    status.empty()
    return timeline


# ─────────────────────────── Streamlit UI ───────────────────────────
st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭", layout="centered")
st.title("🎭 Facial Emotion Recognition")
st.caption("EfficientNet-B3 + CBAM | Graduation Project")

mode = st.radio(
    "اختر طريقة الإدخال:",
    ["📷 كاميرا", "🖼️ رفع صورة", "🎥 تحليل فيديو مسجل"],
    horizontal=True,
)

st.divider()

if mode == "📷 كاميرا":
    st.info("اضغط على زر الكاميرا وامنح المتصفح الإذن، ثم التقط صورة. تقدر تعيد الالتقاط في أي وقت للحصول على تحليل جديد.")
    camera_image = st.camera_input("التقط صورة لوجهك")

    if camera_image is not None:
        model = get_model()
        file_bytes = np.frombuffer(camera_image.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        analyze_and_display(model, img_bgr)

elif mode == "🖼️ رفع صورة":
    uploaded_file = st.file_uploader("ارفع صورة تحتوي على وجه", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        model = get_model()
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        analyze_and_display(model, img_bgr)

else:
    st.info("ارفع فيديو (mp4 يفضّل). للفيديوهات القصيرة اللي فيها تغيّر مشاعر سريع، استخدم "
            "'تتبع دقيق' عشان يلقط كل تغيّر مع فيديو معلّم بالمربع.")

    interval_choice = st.selectbox(
        "طريقة التحليل:",
        ["🔍 تتبع دقيق (فيديو قصير، كل تغيّر)", "15 ثانية", "30 ثانية", "دقيقة واحدة", "دقيقتين"],
        index=0,
    )

    video_file = st.file_uploader("ارفع ملف الفيديو", type=["mp4", "mov", "avi", "mkv"])

    if video_file is not None:
        model = get_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        try:
            if interval_choice.startswith("🔍"):
                target_fps = st.slider("عدد اللقطات بالثانية (دقة التتبع):", 2, 15, 8)
                frames_rgb, change_log = analyze_video_dense(model, tmp_path, target_fps=target_fps)

                if not frames_rgb:
                    st.warning("لم يتم استخراج أي لقطات من الفيديو.")
                else:
                    st.subheader("🎬 Video Analysis by Frames")
                    gif_bytes = frames_to_gif_bytes(frames_rgb, fps=target_fps)
                    st.image(gif_bytes, use_container_width=True)

                    st.subheader("📋 سجل تغيّر المشاعر")
                    if not change_log:
                        st.write("لم يتم رصد أي تغيّر في المشاعر خلال الفيديو.")
                    for timestamp_str, label, confidence in change_log:
                        if label == "لا يوجد وجه":
                            st.write(f"**{timestamp_str}** — لا يوجد وجه ظاهر")
                        else:
                            emoji = EMOTION_EMOJI.get(label, "")
                            st.write(f"**{timestamp_str}** ← {emoji} **{label}** ({confidence*100:.0f}%)")
            else:
                interval_map = {"15 ثانية": 15, "30 ثانية": 30, "دقيقة واحدة": 60, "دقيقتين": 120}
                interval_seconds = interval_map[interval_choice]
                timeline = analyze_video(model, tmp_path, interval_seconds)

                if not timeline:
                    st.warning("لم يتم استخراج أي لقطات من الفيديو.")
                else:
                    st.subheader("📋 الخط الزمني للمشاعر")
                    for timestamp_str, label, confidence, thumb_rgb in timeline:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(thumb_rgb, use_container_width=True)
                        with col2:
                            if label == "لا يوجد وجه":
                                st.write(f"**{timestamp_str}** — لا يوجد وجه ظاهر بهذه اللحظة")
                            else:
                                emoji = EMOTION_EMOJI.get(label, "")
                                st.write(f"**{timestamp_str}** — {emoji} {label} ({confidence*100:.0f}%)")
                        st.divider()
        finally:
            os.remove(tmp_path)
