# Import required libraries
import cv2
import dlib
import keyboard
import time
import sys
from playsound import playsound
from pygame import mixer
from scipy.spatial import distance

# Function to calculate eye-aspect-ratio
def ear_calculation(coords):
    line1 = distance.euclidean(coords[1], coords[5])
    line2 = distance.euclidean(coords[2], coords[4])
    line3 = distance.euclidean(coords[0], coords[3])

    return (line1 + line2) / (2 * line3)

# Initialize alarm track
mixer.init()
mixer.music.load("C:\Users\yashr\Home\Python\DriveX\alarm_tone.mp3") # FIX PATH AS NEEDED
alreadyPlaying = False
startCounter = False
drowsy = False
bot = True  

# Initialize openCV video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize DLIB
detectorInit = dlib.get_frontal_face_detector()
dlibFaceLandmark = dlib.shape_predictor("C:\Users\yashr\Home\Python\Face_Detection\shape_predictor_68_face_landmarks.dat") # FIX PATH AS NEEDED

# Collect user EAR
EAR = 0

name = input("\nHello, Thanks for choosing DriveX\nYou can exit any video window by simply pressing [q]\nPlease start by entering you name: ")
name = name.lower()

print("\nHello", name)

file = open("C:\Users\yashr\Home\Python\DriveX\earInfo.txt", 'r') # FIX PATH AS NEEDED
for line in file:
    splitLine = line.split()
    if splitLine[0] == (name+":"):
        print("Your information has been found!")
        EAR = float(splitLine[1])
file.close()

if EAR == 0:
    inputMethod = int(input("Looks like you are a new user, would you like to test your EAR (eye aspect ratio) [1] or"
        +" do you already know your EAR [2]\nEnter your selection (1 or 2): "))
    if inputMethod == 1:
        sys.exit("\nTo calculate your EAR, please use EARCalculator.py\n")
    EAR = input("Enter you EAR here: ")

    file = open("C:\Users\yashr\Home\Python\DriveX\earInfo.txt", 'a') # FIX PATH AS NEEDED
    file.write(name+ ": "+ str(EAR)+ "\n")
    file.close()

while True:
    _, frame = cap.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detectorInit(grayImage)
    for face in faces:
        face_landmarks = dlibFaceLandmark(grayImage, face)

        # Landmark both eyes
        leftEyeCoords = []
        rightEyeCoords = []

        for z in range(36, 42):
            point = z + 1
            if (z == 41):
                point = 36
            x = face_landmarks.part(z).x
            y = face_landmarks.part(z).y
            x2 = face_landmarks.part(point).x
            y2 = face_landmarks.part(point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
            leftEyeCoords.append((x, y))

        for z in range(42, 48):
            point = z + 1
            if (z == 47):
                point = 42
            x = face_landmarks.part(z).x
            y = face_landmarks.part(z).y
            x2 = face_landmarks.part(point).x
            y2 = face_landmarks.part(point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
            rightEyeCoords.append((x, y))

        for z in range(68):
            x = face_landmarks.part(z).x
            y = face_landmarks.part(z).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

        # Calculate average EAR
        leftEyeRatio = ear_calculation(leftEyeCoords)
        rightEyeRatio = ear_calculation(rightEyeCoords)

        ratioFinal = (leftEyeRatio + rightEyeRatio) / 2
        ratioFinal = round(ratioFinal, 3)

        # Initialize and activate timer
        if (ratioFinal < float(EAR) and not startCounter):
            startCounter = True
            t0 = time.time()

        if startCounter:
            if (time.time() - t0 >= 3) and not alreadyPlaying:
                mixer.music.play()
                drowsy = True
                alreadyPlaying = True

        if (ratioFinal >= float(EAR) and startCounter):
            mixer.music.stop()
            alreadyPlaying = False
            startCounter = False
            drowsy = False

        # On-screen drowsiness warning
        if drowsy:
            cv2.putText(frame, "Drowsy", (250, 100), cv2.FONT_ITALIC, 3, (0, 0, 0), 4)

    cv2.imshow("Face Landmarks", frame)
    cv2.waitKey(1)

    # Exit on 'q'
    if keyboard.is_pressed('q'):
        break

# Release cv window
cap.release()
cv2.destroyAllWindows
