from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup for language model and prompt template
template = """
You are a personal ai assistant assisting a blind person.
Answer the questions asked without any assumptions or unnecessary conversations.

Here is the conversation history(if any): {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="llama3.2:1b")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Global variable to hold the conversation history
context = ""

@app.route('/')
def home():
    return open('index.html').read()

@app.route("/chat", methods=["POST"])
def chat():
    global context
    user_input = request.json.get("question")
    if not user_input:
        return jsonify({"error": "No question provided"}), 400
    
    # Print the question asked by the user
    print(f"User: {user_input}")
    
    # Get the chatbot's response
    result = chain.invoke({"context": context, "question": user_input})
    
    # Print the response from the chatbot
    print(f"AI: {result}")
    
    # Update the conversation history
    context += f"\nUser: {user_input}\nAI: {result}"
    
    return jsonify({"answer": result})

if __name__ == "__main__":
    # Start the Flask app locally
    app.run(debug=True, host="0.0.0.0", port=5000)
