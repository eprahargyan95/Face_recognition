import cv2
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="5BrGlZj9T3HsNh7NX6wy")
project = rf.workspace().project("tesis-1xzsh")
model = project.version("1").model

# Capture video from the laptop's camera
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame to a temporary file
    cv2.imwrite('temp_frame.jpg', frame)

    # Predict on the saved frame
    prediction = model.predict("temp_frame.jpg")

    # Debug print to check the structure of prediction
    print(prediction)

    # Check if 'predictions' key is present
    if 'predictions' in prediction:
        for pred in prediction['predictions']:
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            name = pred.get('class', 'Unknown')
            # Draw rectangle around the detected face
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
            # Put name text above the rectangle
            cv2.putText(frame, name, (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        print("No predictions key in the response")

    cv2.imshow('Frame', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
