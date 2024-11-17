import pickle
import cv2
import os
import numpy as np
import imutils
from deepface import DeepFace

BASE_DIR = os.path.dirname(__file__)
protoPath = os.path.join(BASE_DIR, "face_detection_model/deploy.prototxt")
modelPath = os.path.join(BASE_DIR, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedding_model = os.path.join(BASE_DIR, 'openface_nn4.small2.v1.t7')
embedder = cv2.dnn.readNetFromTorch(embedding_model)
recognizer_file = os.path.join(BASE_DIR, 'output/recognizer.pickle')
le_file = os.path.join(BASE_DIR, 'output/le.pickle')
recognizer = pickle.loads(open(recognizer_file, "rb").read())
le = pickle.loads(open(le_file, "rb").read())

emoji_dict = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😞",
    "surprise": "😮",
    "neutral": "😐"
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open video device")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Could not read frame from camera")
        break
    
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.45:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Emotion detection using DeepFace
            if frame_count % 10 == 0:
                try:
                    analysis = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    dominant_emotion = analysis.get("dominant_emotion", "neutral")
                    emoji = emoji_dict.get(dominant_emotion, "")
                except Exception as e:
                    dominant_emotion = "unknown"
                    emoji = ""
                    print(f"Emotion detection error: {e}")

            # Face recognition using pre-trained recognizer
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Display the results
            text = f"{name}: {proba * 100:.2f}% {emoji}"
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
