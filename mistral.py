import requests
import json
import re
import time

start = time.time()
def extract_json_from_text(text, model="mistral"):
    url = "http://localhost:11434/api/generate"
    
    system_instruction = """
    Bạn là AI trích xuất dữ liệu từ tiếng Việt và trả về JSON đúng định dạng như sau:
    {
    "action": "Thêm"
    "productName": "Tên sản phẩm",
    "categoryName": "Tên loại sản phẩm",
    "brandName": "Tên thương hiệu",
    "unitName": "Tên và giá trị đơn vị tính nếu có"
    }
    Nếu không có thông tin nào thì dùng giá trị null. Không thêm giải thích.
    """
    prompt = f"{system_instruction}\n\nCâu nhập: {text}"

    response = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })

    raw = response.json()["response"]
    print("Phản hồi gốc:\n", raw)

    # Cố gắng parse JSON nếu có thể
    try:
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        print("⚠️ Không parse được JSON. Cần xử lý thêm.")
        return raw

# ✅ Ví dụ
result = extract_json_from_text("Thêm mỹ phẩm nước tẩy trang của cocoon 50ml")
print("Kết quả JSON:\n", result)
end = time.time()
print(f"⏱️ Thời gian chạy: {end - start} giây")