import os
import time
import cv2
import torch
from ultralytics import YOLO


def main():
    # Check if running on macOS with MPS support
    if not torch.backends.mps.is_available():
        print("Error: This script requires macOS with MPS support (Apple Silicon)")
        return

    print("Using MPS (Apple Silicon) device")
    device = 'mps'

    # Path to the trained model
    MODEL_PATH = 'runs/detect/helmet_train18/weights/best.pt'

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please update the MODEL_PATH variable with the correct path.")
        return

    # Load the YOLO model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Initialize video capture (0 for webcam)
    # You can also specify a video file path instead of 0
    cap = cv2.VideoCapture(0)
    
    # Set target FPS to 30
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Get actual FPS (may not be exactly 30 depending on camera)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera running at {actual_fps} FPS")
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame size: {frame_width}x{frame_height}")
    
    # Initialize FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    print("Starting real-time detection. Press 'q' to quit.")
    
    # Main loop
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Update FPS calculation
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:  # Update FPS every second
            fps = fps_frame_count / elapsed_time
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Run YOLO detection
        results = model.predict(
            source=frame,
            conf=0.35,      # Confidence threshold
            device=device,
            verbose=False   # Suppress verbose output for better performance
        )

        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        # Add FPS information
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Display the frame
        cv2.imshow("YOLOv11 Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")


if __name__ == "__main__":
    main() 