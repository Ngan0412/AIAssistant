
from cryptography.fernet import Fernet
import base64
import hashlib
import getpass

def ask_password_and_decrypt_key(encrypted_api_key_b64: str) -> str:
    """
    Nhập mật khẩu, dùng để giải mã API key đã mã hóa trước đó.

    :param encrypted_api_key_b64: chuỗi API key đã được mã hóa (base64 string)
    :return: chuỗi API key gốc (plaintext)
    """
    try:
        # Nhập mật khẩu (không hiển thị trên màn hình)
        password = getpass.getpass("🔑 Nhập mật khẩu để giải mã API Key: ")

        # Tạo key hợp lệ từ mật khẩu
        key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

        # Khởi tạo Fernet với key từ mật khẩu
        f = Fernet(key)

        # Giải mã chuỗi đã mã hóa
        decrypted_api_key = f.decrypt(encrypted_api_key_b64.encode()).decode()
        print("✅ Giải mã thành công.")
        print(decrypted_api_key)

        return decrypted_api_key

    except Exception as e:
        print("❌ Giải mã thất bại. Có thể mật khẩu sai.")
        raise e
    
encrypted_api_key = "gAAAAABog0FSo-Xam59eQZgtSuFnOmEDi4RLoZwvLX4bbvicn0rVM5814RtRNlTOZskoXuotOGv5eOjxrBglt6qtWu2wFzIQZImosqWm83vjdWul4szncWushwiZs01OMv9GWR-c_O9xx503jFDaOrgEXaE-Rr9wYigVwgqB73jOJpOugC8DM5U="  # bạn dán vào đây

# Giải mã khi chạy
OPENROUTER_API_KEY = ask_password_and_decrypt_key(encrypted_api_key)