# Import required libraries
import cv2
import dlib
import keyboard

# Initialize openCV video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize DLIB
detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("C:\Users\yashr\Home\Python\Face_Detection\shape_predictor_68_face_landmarks.dat") # FIX PATH AS NEEDED

while True:
    _, frame = cap.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Plot landmarks
    faces = detector(grayImage)
    for face in faces:
        face_landmarks = dlib_facelandmark(grayImage, face)

        for n in range(68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

    cv2.imshow("Face Landmarks", frame)
    cv2.waitKey(1)

    # Exit on 'q'
    if keyboard.is_pressed('q'):
        break

# Release cv window
cap.release()
cv2.destroyAllWindows