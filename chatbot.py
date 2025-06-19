from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama' 
)

chat_history = []

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    prompt = data.get("prompt", "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400
    if prompt and not any(m["role"] == "system" for m in chat_history):
        chat_history.append({"role": "system", "content": prompt})

    chat_history.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gemma3:4b",
            messages=chat_history,
            stream=False
        )
        reply = response.choices[0].message.content
        
        chat_history.append({"role": "assistant", "content": reply})
        for chunk in chat_history:
            print(chunk)
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
