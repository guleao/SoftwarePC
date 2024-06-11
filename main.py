from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Armazena a posição y do dedo indicador no último quadro
last_index_y = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data):
    global last_index_y

    # Converta a imagem de base64 para um numpy array
    image = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), -1)
    print(image.shape)
    cv2.imshow('image', image); cv2.waitKey(1)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtenha a posição y do dedo indicador
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Compare com a posição y no último quadro
            if last_index_y is not None:
                if index_y > last_index_y:
                    emit('index_up', {})
                elif index_y < last_index_y:
                    emit('index_down', {})
            last_index_y = index_y

    # Converta a imagem processada de volta para base64
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    emit('image', encoded_image)
    print(encoded_image)

if __name__ == '__main__':
    socketio.run(app)

if image is not None:
    print(image.shape)
else:
    print("Image is None")

