from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
from datetime import datetime
from keras.models import load_model
import csv

app = Flask(__name__)


def log_attendance(predicted_class, logged_predictions):
    if predicted_class not in logged_predictions:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%d/%m/%Y")
        

        with open('attendance_log.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_date, current_time, predicted_class])

        logged_predictions.add(predicted_class)
        print(logged_predictions)

def run_face_recognition(model_path):
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    logged_predictions = set()

    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            padding = 50
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            roi = frame[max(0, face_center_y - h // 2 - padding):min(frame.shape[0], face_center_y + h // 2 + padding),
                        max(0, face_center_x - w // 2 - padding):min(frame.shape[1], face_center_x + w // 2 + padding)]
            resized_face = cv2.resize(roi, (200, 400))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            processed_image = resized_face / 255.0
            prediction = model.predict(np.array([processed_image]))
            class_labels = ['J004-Jay Ajmera','J011-Savali Chavan','J012-Snehee Cheeda',
                            'J015 -Farid Damania','J016-Aditya Das','J021- Heet Dhandukia',
                            'J023-Anish Gharat','J024-Suchetan Ghosh','J025-Monish Gosar','J031- Naitik Jain',
                            'J037-Rudra Joshi','J056-Atharva Rode','J058-Jash Shah','J065- Kallind Soni',
                            'J066- Naman Upadhayay','J069- Ismail Wangde','J074-Mihir Shah']
            predicted_class = class_labels[np.argmax(prediction)]
            log_attendance(predicted_class, logged_predictions)
            annotation = predicted_class
            cv2.putText(frame, annotation, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection and Prediction', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/use_model')
def use_model():
    model_name = request.args.get('model')
    model_path = f"{model_name}.h5"
    run_face_recognition(model_path)
    return jsonify({"message": f"Using {model_name}"})

if __name__ == "__main__":
    app.run(debug=True)