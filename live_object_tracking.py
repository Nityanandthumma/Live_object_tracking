import cv2
from ultralytics import YOLO

# Load YOLOv8 model with tracking support
model = YOLO("yolov8s.pt")  # Use yolov8n.pt for faster, or yolov8m/x.pt for more accuracy

# Open default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

print("✅ Live Object Detection + Tracking Started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    # Perform object detection + tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Display result
    cv2.imshow("Live Object Tracking - YOLOv8", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
