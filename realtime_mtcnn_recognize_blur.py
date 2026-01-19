import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from collections import deque

# =============================
# CONFIGURATION
# =============================
KNOWN_DIR = "known_faces"
CAMERA_INDEX = 0

# Blur
BLUR_KERNEL = (45, 45)

# Recognition
KNOWN_THRESHOLD = 0.72
STABILITY_FRAMES = 7

# =============================
# HELPERS
# =============================
def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x) + eps)

def cosine_distance(a, b):
    return 1.0 - np.dot(a, b)

def blur_region(frame, x1, y1, x2, y2):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(
        frame[y1:y2, x1:x2], BLUR_KERNEL, 0
    )

def box_key(box, grid=50):
    x1, y1, x2, y2 = box
    return (x1 // grid, y1 // grid, x2 // grid, y2 // grid)

def preprocess_face(pil_face):
    pil_face = pil_face.resize((160, 160))
    face = np.array(pil_face).astype(np.float32) / 255.0
    face = torch.from_numpy(face).permute(2, 0, 1)
    return face

# =============================
# MODELS
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

mtcnn = MTCNN(
    image_size=160,
    margin=14,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    device=device
)

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# =============================
# LOAD KNOWN FACES
# =============================
known_embeddings = []
known_names = []

def load_known_face(path):
    img = Image.open(path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        return None

    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device))

    emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
    return l2_normalize(emb)

for file in os.listdir(KNOWN_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(file)[0].replace("_", " ").title()
        emb = load_known_face(os.path.join(KNOWN_DIR, file))
        if emb is not None:
            known_embeddings.append(emb)
            known_names.append(name)
            print(f"Loaded known face: {name}")

known_embeddings = np.array(known_embeddings)
print("\nKnown people:", known_names)
print("Starting webcam... Press 'q' to quit\n")

# =============================
# TEMPORAL MEMORY
# =============================
face_history = {}

# =============================
# WEBCAM LOOP
# =============================
cap = cv2.VideoCapture(CAMERA_INDEX)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, _ = mtcnn.detect(pil_img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            key = box_key((x1, y1, x2, y2))

            face_crop = pil_img.crop((x1, y1, x2, y2))
            face_tensor = preprocess_face(face_crop)

            with torch.no_grad():
                emb = resnet(face_tensor.unsqueeze(0).to(device))

            face_emb = l2_normalize(
                emb.squeeze(0).cpu().numpy().astype(np.float32)
            )

            dists = [cosine_distance(face_emb, k) for k in known_embeddings]
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]

            if key not in face_history:
                face_history[key] = deque(maxlen=STABILITY_FRAMES)

            if best_dist < KNOWN_THRESHOLD:
                face_history[key].append(known_names[best_idx])
            else:
                face_history[key].append("Unknown")

            votes = list(face_history[key])
            final = max(set(votes), key=votes.count)

            if final != "Unknown" and votes.count(final) >= STABILITY_FRAMES - 1:
                label = final
                color = (0, 255, 0)
            else:
                blur_region(frame, x1, y1, x2, y2)
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    cv2.imshow("Stable Face Recognition (Unknown Blurred)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
