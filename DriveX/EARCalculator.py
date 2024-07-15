import cv2
import dlib
import keyboard
from scipy.spatial import distance

def __RATIO__ (coords):
    len1 = distance.euclidean(coords[1], coords[5])
    len2 = distance.euclidean(coords[2], coords[4])
    len3 = distance.euclidean(coords[0], coords[3])

    ratio = ((len1 + len2) / (2 * len3))
    return ratio

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("C:/Users/Fnu Yash/Desktop/Python/Face_Detection/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(grayImage)
    for face in faces:
        face_landmarks = dlib_facelandmark(grayImage, face)

        leftEyeCoords = []
        rightEyeCoords = []

        for n in range (36, 42):
            point = n + 1
            if (n == 41):
                point = 36
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            x2 = face_landmarks.part(point).x
            y2 = face_landmarks.part(point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
            leftEyeCoords.append((x, y))

        for n in range(42, 48):
            point = n + 1
            if (n == 47):
                point = 42
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            x2 = face_landmarks.part(point).x
            y2 = face_landmarks.part(point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
            rightEyeCoords.append((x, y))

        for n in range(68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

        leftEyeRatio = __RATIO__(leftEyeCoords)
        rightEyeRatio = __RATIO__(rightEyeCoords)

        ratioFinal = (leftEyeRatio + rightEyeRatio) / 2
        ratioFinal = round(ratioFinal, 3)

        cv2.putText(frame, str(ratioFinal), (250, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)

    cv2.imshow("Face Landmarks", frame)
    cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        break

cap.release()
cv2.destroyAllWindows