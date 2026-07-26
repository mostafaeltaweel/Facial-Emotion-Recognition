import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def face_to_tensor(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(gray, 5, 50, 50)
    image = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)
    mean = np.mean(image)
    gamma = 1.5 if mean < 100 else 0.8 if mean > 155 else 1.2
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    image = cv2.LUT(image, table)
    image = cv2.resize(image, (224, 224))
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return transform(Image.fromarray(rgb))
