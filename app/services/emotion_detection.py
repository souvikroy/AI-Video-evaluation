import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import tensorflow as tf


settings = {'scaleFactor': 1.3, 'minNeighbors': 5, 'minSize': (50, 50)}
labels = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']

face_detection = cv2.CascadeClassifier(r'app/models/haar_cascade_face_detection.xml')
model = tf.keras.models.load_model(r'app/models/network-5Labels.h5')


def EmotionDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected = face_detection.detectMultiScale(gray, **settings)
    emotion = 'Neutral'
    for x, y, w, h in detected:
        cv2.rectangle(image, (x, y), (x + w, y + h), (245, 135, 66), 2)
        cv2.rectangle(image, (x, y), (x + w // 3, y + 20), (245, 135, 66), -1)
        face = gray[y + 5:y + h - 5, x + 20:x + w - 20]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0

        predictions = model.predict(np.array([face.reshape((48, 48, 1))])).argmax()
        emotion = labels[predictions]
    return emotion
