from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

app.secret_key = 'supersecretkey'  # Set a secret key for session management

model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Ensure emotion.npy exists
if not os.path.exists("emotion.npy"):
    np.save("emotion.npy", np.array([""]))

@app.route('/api/capture_emotion', methods=['POST'])
def api_capture_emotion():
    capture_emotion()
    emotion = np.load("emotion.npy")[0]
    return jsonify({"emotion": emotion})
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    lang = request.json.get('language')
    singer = request.json.get('singer')
    emotion = np.load("emotion.npy")[0]
    if not emotion:
        return jsonify({"warning": "Please let me capture your emotion first"})
    else:
        # Construct recommendation URL
        search_query = f"{lang} {emotion} song {singer}"
        url = f"https://www.youtube.com/results?search_query={search_query}"
        np.save("emotion.npy", np.array([""]))
        return jsonify({"url": url})


def capture_emotion():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        res = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)  # Append zeros if landmarks are missing

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)  # Append zeros if landmarks are missing

        lst = np.array(lst).reshape(1, -1)
        pred = label[np.argmax(model.predict(lst))]
        np.save("emotion.npy", np.array([pred]))
        break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
