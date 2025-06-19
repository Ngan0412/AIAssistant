from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Cho phép gọi từ frontend khác domain (CORS policy)

@app.route('/api/message', methods=['POST'])
def receive_message():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message field'}), 400

    user_message = data['message']

    # Xử lý đơn giản: trả về message kèm phản hồi
    response = {
        'message': user_message,
        'response': f"Bạn vừa gửi: {user_message}"
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
