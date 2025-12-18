from mtcnn import MTCNN
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import warnings
import cv2
import os

warnings.filterwarnings("ignore")


MTCNN_OUTPUT_DIR = "results/mtcnn"


plt.style.use('dark_background')


def FaceDetection(image=None, model=None, color='red', url=None, size=10):

    print("Face Detection using MTCNN\n")

    if model is None:
        model = MTCNN()

    if url is not None:
        img = io.imread(url)
    elif image is not None:
        img = np.array(Image.open(image).convert("RGB"))
    else:
        raise ValueError("No image provided")

    coordinates = model.detect_faces(img)

    plt.figure(figsize=(12,6))

    # ---- Bounding boxes ----
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Face Detection")
    plt.axis('off')
    axes = plt.gca()

    for c in coordinates:
        x, y, w, h = c['box']
        rect = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=2)
        axes.add_patch(rect)

    # ---- Keypoints ----
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.title("Keypoints")
    plt.axis('off')

    for c in coordinates:
        for point in c['keypoints'].values():
            plt.scatter(point[0], point[1], c='red', s=size)

    plt.show()
    print("-" * 50)

    os.makedirs(MTCNN_OUTPUT_DIR ,exist_ok=True)

    # Convert RGB -> BGR for OpenCV
    save_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

    for c in coordinates:
        x, y, w, h = c['box']
        cv2.rectangle(save_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for p in c['keypoints'].values():
            cv2.circle(save_img, p, 2, (0, 0, 255), -1)

    filename = os.path.basename(image)
    save_path = os.path.join(MTCNN_OUTPUT_DIR, filename)

    cv2.imwrite(save_path, save_img)
    print(f"[MTCNN][IMAGE] Saved to: {save_path}")



# --------------------------------------------------
# VIDEO FACE DETECTION (MP4 / AVI)
# --------------------------------------------------
def FaceDetectionVideo(video_path):

    print("Face Detection using MTCNN (Video)\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("results/mtcnn", exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"results/mtcnn/{video_name}_mtcnn.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = MTCNN()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        for f in faces:
            x, y, w, h = f['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

            for p in f['keypoints'].values():
                cv2.circle(frame, p, 2, (0, 0, 255), -1)

        cv2.imshow("MTCNN Video Face Detection", frame)
        out.write(frame)   

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[MTCNN][VIDEO] Saved to: {output_path}")



# -------- RUN --------
imagelist = [
    'dataset/1.jpg',
    'dataset/2.png',
    'dataset/3.png',
    'dataset/4.jpg',
    'dataset/classroom.mp4'
]

for path in imagelist:
    path = path.lower()

    if path.endswith(('.jpg', '.png', '.jpeg')):
        FaceDetection(image=path)

    elif path.endswith(('.mp4', '.avi', '.mov')):
        FaceDetectionVideo(path)



    

                           