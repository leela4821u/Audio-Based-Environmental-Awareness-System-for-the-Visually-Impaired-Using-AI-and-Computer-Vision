import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import pyttsx3
import threading
import time

# Set Cache Directory for Torch
os.environ['TORCH_HOME'] = r'F:\MLA\torch_cache'

# Force GPU usage
device = torch.device("cuda")
print(f"Using device: {device}")

# Initialize TTS Engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1)
engine.setProperty('voice', engine.getProperty('voices')[1].id)

# Load MiDaS Model
model_type = "DPT_Hybrid"
midas = torch.hub.load('intel-isl/MiDaS', model_type)
midas.to(device).eval()
transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform

# Load YOLOv8 Model
yolo_model = YOLO("yolov8s.pt")
yolo_model.to(device)

# Load Llama Model
model = OllamaLLM(model="llama3.2:1b")

# Prompts
system_general_message = SystemMessagePromptTemplate.from_template(
    """You are a personal AI assistant for a blind person. Provide precise and helpful responses to their queries without assumptions or unnecessary details.
        you have to respond or chat normally like a real person if asked general questions or having a general conversation."""
)

system_detection_message = SystemMessagePromptTemplate.from_template(
    "You are assisting a blind person by analyzing their surroundings. Generate warnings about detected objects, prioritizing the objects which are closer, and keep the response concise, friendly, and easy to understand like a paragraph of around 70 words. Start the response with 'Be careful, there are '."
)

general_prompt = ChatPromptTemplate.from_messages([
    system_general_message,
    HumanMessagePromptTemplate.from_template(
        "Conversation History: {context}\nQuestion: {question}\nAdditional Information: {additional_data}\nResponse:"
    )
])

detection_prompt = ChatPromptTemplate.from_messages([
    system_detection_message,
    HumanMessagePromptTemplate.from_template(
        "Surrounding Data: {context}\nWarning:"
    )
])

# Shared Data
latest_detections = {"detections": []}
frame_skip = 1 # Process every 5th frame
frame_count = 0

def process_frame(frame):
    global latest_detections, frame_count
    # Resize frame consistently to 640x480 at the start
    resized_frame = cv2.resize(frame, (640, 480))
    persistent_rgb_frame = resized_frame.copy()  # Default to resized frame
    
    if frame_count % frame_skip != 0:
        frame_count += 1
        return persistent_rgb_frame, None  # Return last frame if skipping

    img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to(device)

    # Depth Estimation
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=resized_frame.shape[:2], mode='bicubic', align_corners=False).squeeze()
    output = prediction.cpu().numpy()
    normalized_output = 100 - ((output - output.min()) / (output.max() - output.min()) * 100)
    depth_map = cv2.applyColorMap(cv2.convertScaleAbs(normalized_output, alpha=255/100), cv2.COLORMAP_INFERNO)

    # Object Detection
    results = yolo_model.predict(resized_frame, conf=0.3, verbose=False)
    detections = []
    frame_center_x = resized_frame.shape[1] // 2

    for result in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = result
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = yolo_model.names[int(cls)]
        object_depths = normalized_output[y1:y2, x1:x2]
        closest_depth = float(np.min(object_depths))

        # Position
        object_center_x = (x1 + x2) // 2
        position = "Center" if abs(object_center_x - frame_center_x) < 100 else "Left" if object_center_x < frame_center_x else "Right"

        detections.append({"label": label, "closest_depth": closest_depth, "position": position})

        # Draw on RGB frame
        label_text = f"{label} ({closest_depth:.1f})"
        cv2.rectangle(persistent_rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(persistent_rgb_frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    latest_detections["detections"] = detections
    frame_count += 1
    return persistent_rgb_frame, depth_map

def generate_context(detections):
    close = [f"{d['label']} is close on the {d['position']}" for d in detections if d["closest_depth"] < 30]
    far = [f"{d['label']} is far on the {d['position']}" for d in detections if d["closest_depth"] >= 30]
    return f"{', '.join(close) + '.' if close else 'No close objects.'} {', '.join(far) + '.' if far else 'No far objects.'}"

def detection_thread():
    global last_rgb_frame  # Declare as global to persist across calls
    cap = cv2.VideoCapture(0)  # Using index 1 as specified
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    last_rgb_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame, depth_frame = process_frame(frame)
        if rgb_frame is not None:
            last_rgb_frame = rgb_frame  # Update only when new frame is processed
        if last_rgb_frame is not None:
            cv2.imshow('RGB Stream', last_rgb_frame)  # Always show 640x480 frame
        if depth_frame is not None:
            cv2.imshow('Depth Stream', depth_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc'
            break

    cap.release()
    cv2.destroyAllWindows()


def handle_conversation():
    print("Chatbot started! Type 'exit' to quit.")
    context = ""  # Maintain conversation history like Code 1

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye.")
            engine.say("Goodbye")
            engine.runAndWait()
            break

        if "infront" in user_input.lower():
            # Use latest_detections instead of JSON file
            context_summary = generate_context(latest_detections["detections"])
            result = detection_prompt | model
            output = result.invoke({"context": context_summary})
        else:
            # General conversation
            result = general_prompt | model
            output = result.invoke({"context": context, "question": user_input, "additional_data": ""})
        
        print("Bot:", output)
        engine.say(output)
        engine.runAndWait()
        context += f"\nUser: {user_input}\nAI: {output}"

if __name__ == "__main__":
    detection_t = threading.Thread(target=detection_thread)
    detection_t.daemon = True  # Make it a daemon so it stops when main thread exits
    detection_t.start()
    handle_conversation()  # Run chatbot in main thread