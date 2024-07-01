import cv2
from inference.models.utils import get_roboflow_model  # Assuming this imports your model utility correctly

# Initialize face cascade classifier
face_ref = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# Roboflow model
model_name = "tesis-1xzsh"
model_version = "1"

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get Roboflow face model (this will fetch the model from Roboflow)
model = get_roboflow_model(
    model_id="{}/{}".format(model_name, model_version),
    api_key="5BrGlZj9T3HsNh7NX6wy"
)

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=3)
    return faces

def drawer_box(frame):
    for (x, y, w, h) in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 0, 0), 4)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If the frame was read successfully, display it
    if ret:
        # Detect faces in the frame
        # drawer_box(frame)
        faces = face_detection(frame)
        
        # Run inference on the frame
        results = model.infer(image=frame,
                              confidence=0.4,
                              iou_threshold=0.35)

        # Process results
        for result in results:
            if result.predictions:  # Check if there are any predictions
                for prediction in result.predictions:
                    class_name = prediction.class_name
                    confidence = prediction.confidence
                    text = f"{class_name} || {confidence:.2f}"

                    # Draw bounding boxes and labels for each detected face
                    for (x, y, w, h) in faces:
                        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 250), 4)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Could not read frame.")
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
