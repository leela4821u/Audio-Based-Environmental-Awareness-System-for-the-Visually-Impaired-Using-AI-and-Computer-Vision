import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO  # Import YOLOv8 package
import time

# Set Cache Directory for Torch
os.environ['TORCH_HOME'] = r'F:\MLA\torch_cache'

# Force GPU usage
device = torch.device("cuda")  # Ensuring GPU is always used
print(f"Using device: {device}")

# Load MiDaS Model for Depth Estimation
model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid (medium accuracy, medium inference speed)
midas = torch.hub.load('intel-isl/MiDaS', model_type)
midas.to(device).eval()

# Load Transformation Pipeline
transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

# Load YOLOv8 Model for Object Detection
yolo_model = YOLO("yolov8s.pt")  # Load YOLOv8 Small model
yolo_model.to(device)

# Initialize Video Capture
cap = cv2.VideoCapture(1)

# FPS Variables
frame_count = 0
start_time = time.time()

# Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize Frame for Faster Processing
    resized_frame = cv2.resize(frame, (640, 480))  # Adjust size if needed

    # Convert and Transform Frame for Depth Estimation
    img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to(device)

    # Predict Depth Map
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=resized_frame.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    # Normalize Depth Map to 0-100
    output = prediction.cpu().numpy()
    normalized_output = 100 - ((output - output.min()) / (output.max() - output.min()) * 100)

    # Apply the Color Map for Depth Map
    depth_map = cv2.applyColorMap(cv2.convertScaleAbs(normalized_output, alpha=255/100), cv2.COLORMAP_INFERNO)

    # Run YOLOv8 for Object Detection
    results = yolo_model.predict(resized_frame, conf=0.3)  # Lower confidence threshold to 0.3

    # Draw Bounding Boxes and Labels
    for result in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = result  # Unpack box details
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        object_label = yolo_model.names[int(cls)]

        # Extract Depth Information for Detected Object
        object_depths = normalized_output[y1:y2, x1:x2]
        mean_depth = np.mean(object_depths)  # Average depth in the bounding box
        min_depth = np.min(object_depths)   # Closest depth (minimum value) in the bounding box

        # Display Object Details
        label_text = f"{object_label}: {conf:.2f}, Mean Depth: {mean_depth:.2f}, Closest Depth: {min_depth:.2f}"
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Display FPS
    cv2.putText(resized_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display Results
    cv2.imshow('Depth Map', depth_map)
    cv2.imshow('Camera Feed with Object Detection', resized_frame)

    # Exit on 'q' Key Press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
