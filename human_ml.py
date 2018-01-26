import cv2

face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def detect_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
