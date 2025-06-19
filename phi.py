import requests
import json
import re
import time

start = time.time()
def extract_json_from_text(text, model="phi"):
    url = "http://localhost:11434/api/generate"

    # Sửa prompt cho đúng và rõ ràng
    system_instruction = """
Bạn là AI trích xuất dữ liệu từ tiếng Việt và trả về JSON đúng định dạng như sau:
{
  "action": "Thêm",
  "productName": "Tên sản phẩm",
  "categoryName": "Tên loại sản phẩm",
  "brandName": "Tên thương hiệu",
  "unitName": "Tên và giá trị đơn vị tính nếu có"
}
Nếu không có thông tin nào thì dùng giá trị null. Không thêm giải thích.
"""

    prompt = f"{system_instruction}\n\nCâu nhập: {text}"

    try:
        response = requests.post(url, json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=60)

        if response.status_code != 200:
            print("❌ Lỗi khi gọi API:", response.text)
            return None

        response_json = response.json()

        if "response" not in response_json:
            print("❌ Phản hồi không chứa 'response'")
            print(response_json)
            return None

        raw = response_json["response"]
        print("Phản hồi gốc:\n", raw)

        # Tách nội dung JSON từ phản hồi
        try:
            json_str = re.search(r'{[\s\S]+}', raw).group()
            result = json.loads(json_str)
            return result
        except:
            print("⚠️ Không thể parse JSON từ phản hồi")
            return raw

    except requests.exceptions.RequestException as e:
        print("❌ Lỗi kết nối:", e)
        return None

# ✅ Gọi thử
result = extract_json_from_text("Thêm mỹ phẩm nước tẩy trang của cocoon 50ml", model="phi")
print("✅ Kết quả JSON:\n", result)
end = time.time()
print(f"⏱️ Thời gian chạy: {end - start:.2f} giây")