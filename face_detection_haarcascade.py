import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

BASE_OUTPUT_DIR = "results/haarcascade"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces_image(image_path):
    img = np.array(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))

    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis("off")
    ax = plt.gca()

    for (x,y,w,h) in faces:
        ax.add_patch(
            plt.Rectangle((x,y), w, h, fill=False, color="red", linewidth=2)
        )

    plt.title(f"Image: {os.path.basename(image_path)}")
   
    filename = os.path.basename(image_path)
    save_path = os.path.join(BASE_OUTPUT_DIR, filename)

    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    plt.close()

    print(f"[HAAR][IMAGE] Saved to: {save_path}")






def detect_faces_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(
        BASE_OUTPUT_DIR, f"{video_name}_haarcascade.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[HAAR][VIDEO] Saving video to: {output_path}")
    print("[HAAR][VIDEO] Press ESC to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

        out.write(frame)
        cv2.imshow("Haar Cascade Video Face Detection", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[HAAR][VIDEO] Saved to: {output_path}")


# -------- RUN --------
imagelist = [
    'dataset/1.jpg',
    'dataset/2.png',
    'dataset/3.png',
    'dataset/4.jpg',
    'dataset/classroom.mp4'
]

for path in imagelist:
    if path.lower().endswith(('.jpg', '.png', '.jpeg')):
        detect_faces_image(path)
    elif path.lower().endswith(('.mp4', '.avi', '.mov')):
        detect_faces_video(path)
