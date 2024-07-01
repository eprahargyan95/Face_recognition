import cv2

face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
camera = cv2.VideoCapture(0)

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=3)
    return faces

def drawer_box(frame):
    for (x, y, w, h) in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 0, 0), 4)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        drawer_box(frame)
        cv2.imshow("CuyFace AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    close_window()

if __name__ == '__main__':
    main()
