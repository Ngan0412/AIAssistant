from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex
from llama_index.readers.database import DatabaseReader
from llama_index.llms.openrouter import OpenRouter
from sqlalchemy import create_engine
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # ✅ dùng cho LangChain với OpenRouter
from langchain_core.retrievers import BaseRetriever

from llama_index.core.schema import Document as LlamaDocument
from langchain_core.documents import Document as LangchainDocument
import uuid
import json
from typing import List
from llama_index.core.retrievers import BaseRetriever as LlamaRetriever
from pydantic import Field

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="keepitreal/vietnamese-sbert")
import os
from cryptography.fernet import Fernet
import base64, hashlib, getpass
# dùng cho câu hỏi logic 
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
sql_db = SQLDatabase.from_uri(
    "mssql+pyodbc://@localhost/BOOK_SHOP_API?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)
# === FASTAPI ===
app = FastAPI()
# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể thay "*" bằng ["http://localhost:5173"] để cụ thể hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# === INPUT ===
class QueryRequest(BaseModel):
    question: str

# === INIT API KEY ===
encrypted_api_key = "gAAAAABog0FSo-Xam59eQZgtSuFnOmEDi4RLoZwvLX4bbvicn0rVM5814RtRNlTOZskoXuotOGv5eOjxrBglt6qtWu2wFzIQZImosqWm83vjdWul4szncWushwiZs01OMv9GWR-c_O9xx503jFDaOrgEXaE-Rr9wYigVwgqB73jOJpOugC8DM5U="
OPENROUTER_API_KEY = None
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
qa_chain = None  # <-- để toàn cục dùng được
sql_chain = None  # <-- để toàn cục dùng được
def check_pass():
    global OPENROUTER_API_KEY
    print("🔑 Nhập mật khẩu để giải mã API Key:")
    # password = getpass.getpass()
    key = base64.urlsafe_b64encode(hashlib.sha256("Ngan412@.".encode()).digest())
    f = Fernet(key)
    OPENROUTER_API_KEY = f.decrypt(encrypted_api_key.encode()).decode()
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# chuyen string sang object
def parse_text_to_dict(text: str) -> dict:
    # Tách chuỗi dựa trên dấu `,` nhưng vẫn giữ nguyên cặp key: value
    fields = [field.strip() for field in text.split(",")]
    data = {}
    for field in fields:
        if ':' in field:
            key, value = field.split(":", 1)
            data[key.strip()] = value.strip()
    return data
# === DB INIT ===
def init_qa_chain():
    global qa_chain, OPENROUTER_API_KEY, sql_chain
    if OPENROUTER_API_KEY is None:
        check_pass()
    connection_string = (
    "mssql+pyodbc://@localhost/BOOK_SHOP_API?"
    "driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    )
    engine = create_engine(connection_string)

    reader = DatabaseReader(engine=engine)
    raw_data = reader.load_data(query= """
    SELECT 
        book.id,
        book.title,
        book.image,
        book.quantity,
        book.price,
        author.name AS author_name,
        category.name AS category_name,
        publisher.name AS publisher_name
    FROM 
        book
    JOIN author ON book.authorid = author.id
    JOIN category ON book.categoryid = category.id
    JOIN publisher ON book.publisherid = publisher.id
    WHERE 
        book.isdeleted = 0;
    """)
    documents = []
    for row in raw_data:
        parsed_data = parse_text_to_dict(row.text)
        content = (
            f"Sách '{parsed_data.get('title')}' là một tác phẩm thuộc thể loại {parsed_data.get('category_name')}, "
            f"được viết bởi {parsed_data.get('author_name')} và xuất bản bởi {parsed_data.get('publisher_name')}. "
            f"Hiện có {parsed_data.get('quantity')} bản trong kho, giá bán {parsed_data.get('price')} VNĐ."
        )
        # print(content)
        extra_info = {
            "id": parsed_data.get("id", ""),
            "title": parsed_data.get("title", ""),
            "author": parsed_data.get("author_name", ""),
            "category": parsed_data.get("category_name", ""),
            "price": parsed_data.get("price", ""),
            "image": parsed_data.get("image", "")
        }

        doc = LlamaDocument(
        text=content,
        doc_id=parsed_data.get("id", str(uuid.uuid4())),
        extra_info=extra_info
        )
        documents.append(doc)

    # --- TẠO LLM CHO LLamaIndex ---
    llama_llm =  OpenRouter(
        api_key=OPENROUTER_API_KEY,
        max_tokens=100,
        context_window=4096,
        model="openai/gpt-4o",
    )

    # --- TẠO INDEX ---
    index = VectorStoreIndex.from_documents(
        documents,
        llm=llama_llm,
        llm_predictor=llama_llm,
        embed_model=embed_model
    )

    class CustomRetriever(BaseRetriever):
        llama_retriever: LlamaRetriever = Field()  # ✅ Khai báo đúng cách

        def _get_relevant_documents(self, query: str) -> List[LangchainDocument]:
            results = self.llama_retriever.retrieve(query)
            return [
                LangchainDocument(page_content=node.get_content(), metadata=node.metadata)
                for node in results
            ]
    llama_retriever = index.as_retriever(similarity_top_k=3)
    langchain_retriever = CustomRetriever(llama_retriever=llama_retriever)
    # --- LangChain LLM ---
    langchain_llm = ChatOpenAI(
        base_url=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
        model="openai/gpt-4o",
        max_tokens=100
    )
   
    # --- KẾT NỐI VỚI LANGCHAIN ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=langchain_llm,
        retriever=langchain_retriever,
        return_source_documents=True
    )
def is_logic_question(question: str) -> bool:
    keywords = [
        "đắt nhất", "rẻ nhất", "bao nhiêu quyển", "còn hàng", "giá", "số lượng", "thuộc thể loại", 
        "tác giả nào", "nhà xuất bản nào", "lọc theo", "sắp xếp", "tìm sách"
    ]
    return any(kw in question.lower() for kw in keywords)

def init_logictic_search():
    global sql_chain
    langchain_llm = ChatOpenAI(
        base_url=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
        model="openai/gpt-4o",
        max_tokens=100
    )
    # Tạo PromptTemplate không sinh cú pháp markdown
    custom_prompt = PromptTemplate.from_template("""
    Viết truy vấn SQL CHUẨN CÚ PHÁP SQL SERVER
    Bạn là một trợ lý SQL sử dụng SQL Server.
    Cơ sở dữ liệu có các bảng và cột sau:
    NẾU HỎI VỀ SÁCH THÌ LẤY THÔNG TIN: Id, Title,Image, Price
    Bảng [Book]:
    - Id (uniqueidentifier)  
    - Isbn (char) 
    - Title (nvarchar)                                                                                                                             
    - CategoryId (uniqueidentifier)
    - AuthorId (uniqueidentifier)
    - PublisherId (uniqueidentifier)
    - YearOfPublication (smallint)
    - Price (decimal)
    - Quantity (int)
    - IsDeleted (bit)

    Bảng [Category]:
    - Id (uniqueidentifier)
    - Name (nvarchar(100))
    - IsDeleted (bit)

    Bảng [Publisher]:
    - Id (uniqueidentifier)
    - Name (nvarchar(100))
    - IsDeleted (bit)

    Bảng [Author]:
    - Id (uniqueidentifier)
    - Name (nvarchar(100))
    - IsDeleted (bit)
                                                 
    Ghi chú:
    - Các bảng có quan hệ như sau:
    + Book.CategoryId → Category.Id
    + Book.PublisherId → Publisher.Id
    + Book.AuthorId → Author.Id
    
    TUYỆT ĐỐI KHÔNG được thêm bất kỳ dấu markdown nào như dấu ``` hoặc dấu `.
    KHÔNG BAO GIỜ được viết:
    ```sql                                           .
    Chỉ trả về đúng câu lệnh sql
    Câu hỏi: {input}
    SQL:
    """)

    sql_chain = SQLDatabaseChain.from_llm(
    llm=langchain_llm,
    db=sql_db,
    prompt=custom_prompt,
    verbose=True,
    )

# === GỌI CHAT ===
@app.post("/chat")
def ask_bookshop(request: QueryRequest):
    if qa_chain is None:
        init_qa_chain()

    question = request.question
    products = []
    suggestions = []
    # Nếu là câu hỏi logic → chạy SQLChain
    if is_logic_question(question):
        try:
            init_logictic_search()
            
            response = sql_chain.run(question)
           
            connection_string = (
            "mssql+pyodbc://@localhost/BOOK_SHOP_API?"
            "driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
            )
            engine = create_engine(connection_string)

            reader = DatabaseReader(engine=engine)
            raw_data = reader.load_data(query=response)
            for row in raw_data:
                parsed_data = parse_text_to_dict(row.text)
                products.append(parsed_data)
            return {
                "products": products,
                "suggestions": suggestions
            }
        except Exception as e:
                raise e

    # Ngược lại, dùng semantic search + gợi ý hình ảnh
    result = qa_chain.invoke({"query": question})
    try:
        products = json.loads(result["result"])
    except Exception:
        products = result["result"]

    source_docs = result.get('source_documents', [])
    for doc in source_docs:
        metadata = doc.metadata
        suggestions.append({
            "id": metadata.get("id"),
            "title": metadata.get("title"),
            "image": metadata.get("image"),
            "price": metadata.get("price"),
        })

    return {
        "products": products,
        "suggestions": suggestions
    }
# === MAIN ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ChatBotLLmaIndex:app", host="127.0.0.1", port=8000, reload=False)
