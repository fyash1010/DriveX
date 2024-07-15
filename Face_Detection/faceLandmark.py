import cv2
import dlib
import keyboard

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("C:/Users/Fnu Yash/Desktop/Python/Face_Detection/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(grayImage)
    for face in faces:
        face_landmarks = dlib_facelandmark(grayImage, face)

        for n in range(68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

    cv2.imshow("Face Landmarks", frame)
    cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        break

cap.release()
cv2.destroyAllWindows