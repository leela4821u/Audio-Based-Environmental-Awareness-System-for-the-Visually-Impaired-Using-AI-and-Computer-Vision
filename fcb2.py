from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import json

# Define system-level and user-level prompts for clarity and modularity
system_general_message = SystemMessagePromptTemplate.from_template(
    "You are a personal AI assistant for a blind person. Provide precise and helpful responses to their queries without assumptions or unnecessary details."
)

system_detection_message = SystemMessagePromptTemplate.from_template(
    "You are assisting a blind person by analyzing their surroundings. Generate warnings about detected objects, prioritizing the objects which are closer, and keep the response concise, friendly, and easy to understand like a paragraph of around 70 words.  Start the response with 'Be careful, there are '."
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

# Define the model
model = OllamaLLM(model="llama3.2")

def load_json_data(json_file):
    """Load JSON data from a file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return "I couldn’t find the detection data. Please make sure the JSON file is available."
    except Exception as e:
        return f"An error occurred while processing the detection data: {e}"

def generate_context_from_data(json_data):
    """Generate a summary of surroundings based on JSON detection data."""
    close_objects = []
    far_objects = []

    # Iterate over the detections and categorize objects
    for item in json_data:
        label = item["label"]
        closest_depth = item["closest_depth"]
        position = item["position"]

        # Classify objects by proximity
        if closest_depth < 30:
            close_objects.append(f"{label} is close and on the {position}")
        else:
            far_objects.append(f"{label} is far and on the {position}")

    # Create summaries for close and far objects
    close_objects_summary = ", ".join(close_objects) + "." if close_objects else "No close objects detected."
    far_objects_summary = ", ".join(far_objects) + "." if far_objects else "No far objects detected."

    # Combine summaries
    return f"{close_objects_summary} {far_objects_summary}"

def handle_conversation():
    """Main conversational handler."""
    context = ""
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        if "infront" in user_input.lower():
            # Load and process detection data
            json_data = load_json_data("detss.json")
            if isinstance(json_data, str):  # Error message
                print("Bot: ", json_data)
                context += f"\nUser: {user_input}\nAI: {json_data}"
            else:
                # Generate context summary from detection data
                context_summary = generate_context_from_data(json_data)

                # Use the detection template
                result = detection_prompt | model
                output = result.invoke({"context": context_summary})
                
                print("Bot: ", output)
                context += f"\nUser: {user_input}\nAI: {output}"
        else:
            # Use the general template for regular conversations
            result = general_prompt | model
            output = result.invoke({
                "context": context,
                "question": user_input,
                "additional_data": ""
            })
            print("Bot: ", output)
            context += f"\nUser: {user_input}\nAI: {output}"

if __name__ == "__main__":
    handle_conversation()
