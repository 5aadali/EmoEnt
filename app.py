from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os

app = Flask(__name__)
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

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        emotion = np.load("emotion.npy")[0]
    except FileNotFoundError:
        emotion = ""

    lang = session.get('language', '')
    singer = session.get('singer', '')

    if request.method == 'POST':
        lang = request.form['language']
        singer = request.form['singer']
        session['language'] = lang
        session['singer'] = singer

        if 'capture' in request.form:
            capture_emotion()
        elif 'recommend' in request.form:
            emotion = np.load("emotion.npy")[0]
            if not emotion:
                warning = "Please let me capture your emotion first"
                return render_template('index.html', warning=warning, language=lang, singer=singer)
            else:
                webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
                np.save("emotion.npy", np.array([""]))
                session.pop('language', None)
                session.pop('singer', None)
                emotion = ""

    return render_template('index.html', emotion=emotion, language=lang, singer=singer)

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
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            np.save("emotion.npy", np.array([pred]))
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
