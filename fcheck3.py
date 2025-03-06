import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO  # Import YOLOv8 package
import json

# Set Cache Directory for Torch
os.environ['TORCH_HOME'] = r'F:\MLA\torch_cache'

# Force GPU usage
device = torch.device("cuda")  # Ensuring GPU is always used
print(f"Using device: {device}")

# Load MiDaS Model for Depth Estimation
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid (medium accuracy, medium inference speed)
midas = torch.hub.load('intel-isl/MiDaS', model_type)
midas.to(device).eval()

# Load Transformation Pipeline
transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform

# Load YOLOv8 Model for Object Detection
yolo_model = YOLO("yolov8s.pt")  # Load YOLOv8 Small model
yolo_model.to(device)

# Load Image
image_path = r'1.jpg'  # Replace with your image file path
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

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

# Run YOLOv8 for Object Detection
results = yolo_model.predict(resized_frame, conf=0.3)  # Lower confidence threshold to 0.3

# Store Results
detections = []
frame_center_x = resized_frame.shape[1] // 2  # Find the center of the frame's width

for result in results[0].boxes.data.cpu().numpy():
    x1, y1, x2, y2, conf, cls = result  # Unpack box details
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    object_label = yolo_model.names[int(cls)]

    # Extract Depth Information for Detected Object
    object_depths = normalized_output[y1:y2, x1:x2]
    mean_depth = np.mean(object_depths)  # Average depth in the bounding box
    min_depth = np.min(object_depths)   # Closest depth (minimum value) in the bounding box

    # Determine object position (left, center, right)
    object_center_x = (x1 + x2) // 2
    if object_center_x < frame_center_x - 100:
        position = "Left"
    elif object_center_x > frame_center_x + 100:
        position = "Right"
    else:
        position = "Center"

    detections.append({
        "label": object_label,
        "confidence": float(conf),
        "bounding_box": [x1, y1, x2, y2],
        "mean_depth": float(mean_depth),
        "closest_depth": float(min_depth),
        "position": position  # Add position data (left, center, or right)
    })

# Save Results to JSON
output_file = "detss.json"
with open(output_file, 'w') as f:
    json.dump(detections, f, indent=4)

print(f"Detection results saved to {output_file}")

# Display Results
for detection in detections:
    x1, y1, x2, y2 = detection['bounding_box']
    label_text = f"{detection['label']}: {detection['confidence']:.2f}, Mean Depth: {detection['mean_depth']:.2f}, Closest Depth: {detection['closest_depth']:.2f}, Position: {detection['position']}"
    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(resized_frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Apply the Color Map for Depth Map
depth_map = cv2.applyColorMap(cv2.convertScaleAbs(normalized_output, alpha=255/100), cv2.COLORMAP_INFERNO)

# Function to Display Depth Value on Cursor Hover
def show_depth(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Check if the mouse is moving
        depth_value = normalized_output[y, x]  # Get depth value at cursor position
        print(f"Depth at ({x}, {y}): {depth_value:.2f}")  # Print depth value to console

# Create Window and Set Mouse Callback
cv2.namedWindow('Depth Map')
cv2.setMouseCallback('Depth Map', show_depth)

while True:
    cv2.imshow('Depth Map', depth_map)
    cv2.imshow('Image with Detections', resized_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
