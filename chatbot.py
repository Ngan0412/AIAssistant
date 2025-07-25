from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
import pyodbc
import re
app = Flask(__name__)
CORS(app)  

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama' 
)

def get_db_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"         
        "DATABASE=BOOK_SHOP_API;"        
        "Trusted_Connection=yes;"      
    )
    return conn

system_prompt = """
You are a helpful assistant. Your task is to convert natural language questions into SQL queries. 
Assume the database has a table called 'Book' with the columns: ID, Isbn, Title, Price, YearOfPublication.
Only output the SQL query without explanation.
"""

def question_to_sql_with_gemma(user_question):
    response = client.chat.completions.create(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()


chat_history = []
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # Check if user is asking for data
    if any(keyword in message.lower() for keyword in ["liệt kê", "sách", "đơn hàng", "sản phẩm", "truy vấn", "ở đâu", "bao nhiêu"]):
        try:
            sql_query_raw = question_to_sql_with_gemma(message)
            print(sql_query_raw)
            
            sql_query = clean_sql_output(sql_query_raw)
            print(sql_query)
            if not sql_query.lower().startswith("select"):
                    return jsonify({"reply": "Bạn hỏi lại đi"}), 200

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(sql_query)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            result = [dict(zip(columns, row)) for row in rows]
            return jsonify({"reply": result})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Nếu không phải truy vấn → chat thông thường
    chat_history.append({"role": "user", "content": message})
    try:
        response = client.chat.completions.create(
            model="gemma3:4b",
            messages=chat_history,
            stream=False
        )
        reply = response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def clean_sql_output(raw_output):
    # Tìm đoạn bắt đầu bằng SELECT
    match = re.search(r"SELECT[\s\S]*?;", raw_output, re.IGNORECASE)
    return match.group(0).strip() if match else ""

if __name__ == '__main__':
    app.run(debug=True)
