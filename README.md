🎥 Real-Time Face Recognition with Unknown Face Blurring

This project implements a real-time face recognition system using MTCNN for face detection and FaceNet (InceptionResnetV1) for face recognition.
Recognized faces are labeled, while unknown faces are automatically blurred for privacy.

The system also includes temporal stability, ensuring that identities are only confirmed after appearing consistently across multiple frames.

🚀 Features

✅ Real-time webcam face detection

✅ Face recognition using deep embeddings

✅ Unknown faces are blurred (privacy-first)

✅ Temporal voting to prevent flickering labels

✅ Supports multiple known identities

✅ CPU & GPU compatible (PyTorch)

🧠 Technologies Used

Python 3.9+

OpenCV

PyTorch

facenet-pytorch

MTCNN

InceptionResnetV1 (VGGFace2 pretrained)

NumPy

PIL (Pillow)

📂 Project Structure
visionprojesi/
│
├── realtime_mtcnn_recognize_blur.py
├── known_faces/
│   ├── tarkan.jpg
│   ├── alice.png
│   └── bob.jpeg
│
├── venv/
└── README.md


known_faces/
Contains reference images of known people.
File name = person name (underscores allowed).

🖼️ How It Works

Face Detection

MTCNN detects all faces in each webcam frame.

Face Embedding

Each detected face is resized to 160×160

A 512-D embedding is extracted using FaceNet.

Recognition

Cosine distance is computed against known embeddings.

If distance < threshold → known face

Otherwise → unknown face

Temporal Stability

Identity must appear consistently across multiple frames.

Prevents false positives and flickering labels.

Privacy Protection

Unknown faces are blurred in real time.

⚙️ Installation
1️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

2️⃣ Install Dependencies
pip install torch torchvision torchaudio
pip install opencv-python facenet-pytorch pillow numpy


⚠️ If you have CUDA installed, PyTorch will automatically use GPU.

▶️ Running the Project
python realtime_mtcnn_recognize_blur.py


Press q to quit

Webcam must be connected

Console will show loaded known identities

🧪 Configuration

You can adjust these values inside the script:

KNOWN_THRESHOLD = 0.72     # Lower = stricter matching
STABILITY_FRAMES = 7      # Frames required for stable identity
BLUR_KERNEL = (45, 45)    # Blur strength

🧩 Adding Known Faces

Add an image to known_faces/

Name the file after the person:

john_doe.jpg  →  John Doe


Restart the program

❗ Common Issues
❌ Webcam Not Opening

Check CAMERA_INDEX = 0

Try 1 or 2 if multiple cameras exist

❌ torch.cat(): expected a non-empty list of Tensors

✔ Already fixed in this version
This project uses MTCNN only once per frame, preventing this error.

🔒 Privacy & Ethics

Unknown individuals are never identified

Faces are anonymized via blurring

No data is stored or transmitted

This makes the system suitable for GDPR-aware applications.

🛠️ Possible Improvements

FPS optimization

Face tracking (Kalman / SORT)

ArcFace or AdaFace embeddings

Mask / sunglasses robustness

Face database persistence