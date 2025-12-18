Face Detection Project (Images & Video)

This project demonstrates face detection on images and videos using two popular approaches:

Haar Cascade (OpenCV)

MTCNN (Multi-Task Cascaded Convolutional Neural Network)

The system processes multiple images and a classroom video, detects faces, and visualizes bounding boxes (and facial keypoints for MTCNN).

📁 Dataset

The dataset consists of:

Portrait images with single faces

A group photo with multiple faces

A classroom video containing multiple people

Supported formats:

Images: .jpg, .png, .jpeg

Videos: .mp4, .avi, .mov

🧠 Methods Used
1. Haar Cascade (OpenCV)

Classical machine-learning approach

Fast and lightweight

Uses grayscale images

Best suited for frontal faces

Implemented in:

face_detection_haarcascade.py 

face_detection_haarcascade

Features

Face detection on images with red bounding boxes

Real-time face detection on video streams

ESC key to stop video processing

2. MTCNN

Deep-learning-based approach

More accurate and robust

Detects facial landmarks (eyes, nose, mouth)

Implemented in:

face_detection_mtcnn.py 

face_detection_mtcnn

Features

Face bounding boxes

Facial keypoint visualization

Image and video support

Better performance on multiple faces and varying angles

🛠️ Requirements

Install the required dependencies:

pip install opencv-python numpy pillow matplotlib mtcnn scikit-image


⚠️ MTCNN may require TensorFlow depending on your environment.

▶️ How to Run
Haar Cascade
python face_detection_haarcascade.py

MTCNN
python face_detection_mtcnn.py


Both scripts automatically process:

All images in the dataset

The classroom video

📊 Output
Image Processing

Displays detected faces with bounding boxes

MTCNN additionally shows facial keypoints

Video Processing

Live video window with face detection

Press ESC to exit

License

This project is for educational and research purposes.