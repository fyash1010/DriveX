# Adapted code for DriveX - Fatigue Detection using SVM and temporal smoothing
# Import required libraries
import cv2
import dlib
import keyboard
import time
import sys
import joblib
import numpy as np
from pygame import mixer
from scipy.spatial import distance
from collections import deque

# EAR Calculation
def ear_calculation(coords):
    line1 = distance.euclidean(coords[1], coords[5])
    line2 = distance.euclidean(coords[2], coords[4])
    line3 = distance.euclidean(coords[0], coords[3])
    return (line1 + line2) / (2.0 * line3)

# Initialize alarm
mixer.init()
mixer.music.load("C:\\Users\\yashr\\Home\\Python\\DriveX\\alarm_tone.mp3")
alreadyPlaying = False

# Load webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\yashr\\Home\\Python\\Face_Detection\\shape_predictor_68_face_landmarks.dat")

# Load SVM model
svm_model = joblib.load("C:\\Users\\yashr\\Home\\Python\\DriveX\\svm_drowsiness_model.pkl")

# Temporal smoothing config
window_size = 15  # frames
prediction_queue = deque(maxlen=window_size)

# Collect user EAR
name = input("\nHello, Thanks for choosing DriveX\nYou can exit any video window by pressing [q]\nPlease enter your name: ").lower()
EAR = 0
with open("C:\\Users\\yashr\\Home\\Python\\DriveX\\earInfo.txt", 'r') as file:
    for line in file:
        splitLine = line.split()
        if splitLine[0] == (name + ":"):
            print("Your information has been found!")
            EAR = float(splitLine[1])

if EAR == 0:
    inputMethod = int(input("Looks like you're new!\nWould you like to test your EAR [1] or enter it manually [2]?\nEnter 1 or 2: "))
    if inputMethod == 1:
        sys.exit("\nTo calculate your EAR, please use EARCalculator.py\n")
    EAR = float(input("Enter your EAR: "))
    with open("C:\\Users\\yashr\\Home\\Python\\DriveX\\earInfo.txt", 'a') as file:
        file.write(name + ": " + str(EAR) + "\n")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        leftEye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        rightEye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Draw eyes and landmarks
        for eye in [leftEye, rightEye]:
            for i in range(len(eye)):
                x1, y1 = eye[i]
                x2, y2 = eye[(i + 1) % len(eye)]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

        # Compute EAR
        leftEAR = ear_calculation(leftEye)
        rightEAR = ear_calculation(rightEye)
        avg_EAR = round((leftEAR + rightEAR) / 2.0, 3)

        # Predict using SVM
        feature_vector = np.array([[avg_EAR]])  # if you used more features in training, include them here
        prediction = svm_model.predict(feature_vector)[0]  # 1 = drowsy, 0 = alert
        prediction_queue.append(prediction)

        # Temporal smoothing via majority voting
        drowsy = prediction_queue.count(1) > (window_size // 2)

        # Show alert if drowsy
        if drowsy:
            cv2.putText(frame, "Drowsy", (250, 100), cv2.FONT_ITALIC, 3, (0, 0, 255), 4)
            if not alreadyPlaying:
                mixer.music.play()
                alreadyPlaying = True
        else:
            if alreadyPlaying:
                mixer.music.stop()
                alreadyPlaying = False

    # Show frame
    cv2.imshow("DriveX - Fatigue Detection", frame)
    if keyboard.is_pressed('q'):
        break

cap.release()
cv2.destroyAllWindows()
