import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model_path = "best_2.pt"  # Update this path if needed
model = YOLO(model_path)

# Choose source: 0 = webcam, or replace with a video path
source = 0  # Use 0 for webcam, or "path/to/video.mp4" for a file

# Open video capture
cap = cv2.VideoCapture(source)

# Set higher resolution for webcam (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Optional: Save output video
save_output = True
output_path = "output_detected_1.mp4"
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ))

# Loop over frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference using YOLO with higher input size
    results = model(frame, conf=0.25, imgsz=800)

    # Annotate results on the frame
    annotated_frame = results[0].plot()

    # Resize frame for display (optional)
    display_frame = cv2.resize(annotated_frame, (1280, 720))

    # Show the frame
    cv2.imshow("YOLOv8 Detection", display_frame)

    # Save the frame to output video
    if save_output:
        out.write(annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()