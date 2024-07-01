import cv2
from inference.models.utils import get_roboflow_model  # Assuming this imports your model utility correctly

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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was read successfully, display it
    if ret:
        # Run inference on the frame
        
        results = model.infer(image=frame,
                              confidence=0.51,
                              iou_threshold=0.5)

        # Process results
        for result in results:
            if result.predictions:  # Check if there are any predictions
                for prediction in result.predictions:
                    x = int(prediction.x)  # Convert float positions to integers
                    y = int(prediction.y)
                    w = int(prediction.width)
                    h = int(prediction.height)
                    class_name = prediction.class_name
                    confidence = prediction.confidence
                    text = class_name + "||" + str(round(confidence,2))
                    xPos = prediction.x
                    yPos = prediction.y
                    posPred = str(xPos) + " " + str(yPos)
                    print(posPred)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x - 100, y - 100), (x + 100, y + 130), (255, 0, 0), 2)
                    cv2.putText(frame, text, (x - 100, y - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

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