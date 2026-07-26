"""
Preprocessing pipeline — MUST match training exactly (notebook CELL 3 / CELL 21),
otherwise accuracy will drop badly at inference time.

Training pipeline was:
  1. Read image as GRAYSCALE
  2. Bilateral filter (noise reduction, keeps edges)
  3. CLAHE (adaptive contrast enhancement)
  4. Gamma correction (auto brightness balancing)
  5. Resize to 224x224
  6. Convert grayscale -> 3-channel RGB (duplicate channel, EfficientNet expects 3 channels)
  7. ToTensor + Normalize with ImageNet mean/std
"""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224

# Same normalization used in test_transforms during training
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(gray_image):
    """
    Applies the exact 3-step enhancement used during training.
    Input: single-channel grayscale numpy image (uint8)
    Output: single-channel grayscale numpy image (uint8), enhanced
    """
    # 1) Bilateral filter — reduces noise while preserving edges
    image = cv2.bilateralFilter(gray_image, 5, 50, 50)

    # 2) CLAHE — adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # 3) Gamma correction — auto brightness balancing
    mean_intensity = np.mean(image)
    if mean_intensity < 100:
        gamma = 1.5
    elif mean_intensity > 155:
        gamma = 0.8
    else:
        gamma = 1.2
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    image = cv2.LUT(image, table)

    return image


def face_crop_to_tensor(face_bgr):
    """
    Full pipeline: BGR face crop (from OpenCV) -> normalized tensor ready for the model.
    Input: face_bgr -> a cropped face region in BGR format (as returned by OpenCV/webcam)
    Output: torch.Tensor of shape (3, 224, 224), NOT batched yet
    """
    # Convert to grayscale (same as training, which read images as IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # Apply the same enhancement steps used in training
    enhanced = preprocess_image(gray)

    # Resize to the training resolution
    resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))

    # Grayscale -> 3-channel RGB (EfficientNet expects 3 channels)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # To PIL -> tensor -> normalize (matches test_transforms exactly)
    pil_image = Image.fromarray(rgb)
    tensor = inference_transform(pil_image)

    return tensor
