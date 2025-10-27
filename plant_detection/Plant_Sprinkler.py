import cv2
import serial  # Import the serial library
import time    # Import the time library
from ultralytics import YOLO

# --- Arduino Communication Setup ---
try:
    # IMPORTANT: Replace 'COM5' with your Arduino's actual serial port.
    # On Linux/Mac, it might be '/dev/ttyACM0' or '/dev/ttyUSB0'.
    arduino_port = 'COM5'
    baud_rate = 9600
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)  # Wait for the connection to establish
    print(f"✅ Connected to Arduino on {arduino_port}")
    arduino_connected = True
except serial.SerialException as e:
    print(f"⚠️ Error: Could not connect to Arduino. {e}")
    print("Running detection without Arduino communication.")
    arduino_connected = False

def send_to_arduino(signal):
    """Sends a single character signal to the Arduino if connected."""
    if arduino_connected:
        try:
            ser.write(signal.encode())  # Send the signal as bytes
            print(f"Sent signal '{signal}' to Arduino.") # Debugging print
        except Exception as e:
            print(f"Failed to send signal to Arduino: {e}")

# --- YOLOv8 and Video Capture Setup ---
model_path = "best_2.pt"
model = YOLO(model_path)
source = 0
cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Optional: Save output video
save_output = False # Disabled by default to prevent saving frozen frames
output_path = "output_detected_1.mp4"
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ))

# --- Main Detection Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference using YOLO
    results = model(frame, conf=0.60, imgsz=800) # Lowered conf threshold to catch 20%+

    delay_duration = 0 # This will hold the freeze time in seconds

    # Check for detections and their confidence
    if len(results[0].boxes) > 0:
        # Get the highest confidence score from all detections in the frame
        max_confidence = max(results[0].boxes.conf).item()

        if max_confidence >= 0.80:
            print(f"Very High confidence ({max_confidence:.2f}). Freezing for 4s.")
            send_to_arduino('4') # Signal for 4-second action
            delay_duration = 4
        elif max_confidence >= 0.60:
            print(f"High confidence ({max_confidence:.2f}). Freezing for 3s.")
            send_to_arduino('3') # Signal for 3-second action
            delay_duration = 3
        elif max_confidence >= 0.40:
            print(f"Medium confidence ({max_confidence:.2f}). Freezing for 2s.")
            send_to_arduino('2') # Signal for 2-second action
            delay_duration = 2
        elif max_confidence >= 0.20:
            print(f"Low confidence ({max_confidence:.2f}). Freezing for 1s.")
            send_to_arduino('1') # Signal for 1-second action
            delay_duration = 1
        # Detections below 0.2 confidence will be ignored

    # Annotate results on the frame
    annotated_frame = results[0].plot()

    # Resize frame for display
    display_frame = cv2.resize(annotated_frame, (1280, 720))

    # Show the frame
    cv2.imshow("YOLOv8 Detection", display_frame)

    # --- FREEZE AND SKIP FRAMES LOGIC ---
    if delay_duration > 0:
        start_time = time.time()
        quit_pressed = False
        
        # While we are in the delay period...
        while time.time() - start_time < delay_duration:
            # Grab and discard frames from the camera buffer to "skip" them
            cap.grab()
            
            # Keep the display window responsive and check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_pressed = True
                break
        
        if quit_pressed:
            break

    # Standard check for quit key if no freeze occurred
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Release Resources ---
print("Releasing resources...")
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()

if arduino_connected:
    ser.close()
    print("Arduino connection closed.")

