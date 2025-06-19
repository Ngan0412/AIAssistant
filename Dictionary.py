import re

def simple_extract(text):
    product = re.search(r"thêm (.*?) của", text)
    brand = re.search(r"của (.*?) (\d+ml|\d+g|ml|g)", text)
    unit = re.search(r"(\d+ml|\d+g|ml|g)", text)

    return {
        "action": "Thêm",
        "productName": product.group(1) if product else None,
        "brandName": brand.group(1) if brand else None,
        "unitName": unit.group(1) if unit else None,
        "categoryName": "mỹ phẩm"  # hardcode hoặc phân loại nâng cao
    }

print(simple_extract("Thêm sữa rửa mặt của cocoon 200ml"))