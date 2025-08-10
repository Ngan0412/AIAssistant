from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex
from llama_index.readers.database import DatabaseReader
from llama_index.llms.openrouter import OpenRouter
from sqlalchemy import create_engine
from langchain.chains import RetrievalQA

from langchain_openai import ChatOpenAI  
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
import numpy as np
# dùng cho câu hỏi logic 
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from llama_index.core.vector_stores  import VectorStoreQuery
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
OPENROUTER_API_KEY = ""
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
qa_chain = None  # <-- để toàn cục dùng được
sql_chain = None  # <-- để toàn cục dùng được
index = None  # <-- để toàn cục dùng được
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
    global qa_chain, OPENROUTER_API_KEY, sql_chain,index
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
        authors.name AS author_name,
        categories.name AS category_name,
        publishers.name AS publisher_name
    FROM 
        books as book
    JOIN authors ON book.authorid = authors.id
    JOIN categories ON book.categoryid = categories.id
    JOIN publishers ON book.publisherid = publishers.id
    WHERE 
        book.isdeleted = 0;
    """)
    documents = []
    for row in raw_data:
        parsed_data = parse_text_to_dict(row.text)
        content = (
            f"Sách '{parsed_data.get("title")}' là một tác phẩm thuộc thể loại {parsed_data.get("category_name")}, "
            f"được viết bởi {parsed_data.get("author_name")} và xuất bản bởi {parsed_data.get("publisher_name")}. "
            f"Hiện có {parsed_data.get("quantity")} bản trong kho, giá bán {parsed_data.get("price")} VNĐ."
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
    Bảng [Books]:
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

    Bảng [Categories]:
    - Id (uniqueidentifier)
    - Name (nvarchar(100))
    - IsDeleted (bit)

    Bảng [Publisheries]:
    - Id (uniqueidentifier)
    - Name (nvarchar(100))
    - IsDeleted (bit)

    Bảng [Authors]:
    - Id (uniqueidentifier)
    - Name (nvarchar(100))
    - IsDeleted (bit)
                                                 
    Bảng [OrderItems]:
    - BookId (uniqueidentifier)
    - Quantity (smallint)
                                                 
    Ghi chú:
    - Các bảng có quan hệ như sau:
    + Books.CategoryId → Categories.Id
    + Books.PublisherId → Publishers.Id
    + Books.AuthorId → Authors.Id
    + OrderItems.BookId → Books.Id
    
    QUY TẮC BẮT BUỘC:
    1. Chỉ trả về CÂU LỆNH SQL DUY NHẤT.
    2. KHÔNG được bao quanh bằng bất kỳ ký tự Markdown nào (ví dụ: ```, `sql`, ```sql).
    3. Không thêm bất kỳ văn bản, giải thích, tiêu đề hoặc ký tự ngoài câu SQL.
    4. Nếu vi phạm các điều trên, câu trả lời coi như sai.

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
                print("rowwwwwwwwwwww")
                print(row)
                parsed_data = parse_text_to_dict(row.text)
                print("rowwwwwwwwwwww2")
                print(parsed_data)
                
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
class RecommendRequest(BaseModel):
    book_ids: list[str]  

from llama_index.core.vector_stores import VectorStoreQuery

@app.post("/recommend")
def recommend_for_user(req: RecommendRequest):
    if qa_chain is None:
        init_qa_chain()

    global index
    if index is None:
        raise ValueError("Vector index chưa được khởi tạo.")

    recommend_books = req.book_ids
    if not recommend_books:
        return {"error": "Danh sách book_ids rỗng"}

    # ===== Lấy vector embedding của từng sản phẩm trong lịch sử =====
    recommend_set = set(map(str, recommend_books))
    vectors = []
    for doc in index.docstore.docs.values():
        doc_id = doc.metadata.get("id")
        if doc_id is not None and str(doc_id) in recommend_set:
            emb = embed_model.get_text_embedding(doc.get_content())
            if emb:
                vectors.append(emb)

    if not vectors:
        return {"error": "Không tìm thấy vector cho sản phẩm lịch sử"}

    # ===== Tính vector trung bình =====
    user_vector = np.mean(vectors, axis=0).tolist()

    # ===== Tạo query object đúng chuẩn =====
    query_obj = VectorStoreQuery(
        query_embedding=user_vector,
        similarity_top_k=8  # lấy nhiều hơn để loại trừ rồi còn đủ kết quả
    )

    # ===== Query vector store =====
    results = index.vector_store.query(query_obj)

    recommended_books = []
    if hasattr(results, "nodes") and results.nodes:
        for node in results.nodes:
            parsed_data = node.metadata
            if parsed_data.get("id") not in recommend_books:  # loại trừ sách đã có
                recommended_books.append({
                    "id": parsed_data.get("id", ""),
                    "title": parsed_data.get("title", ""),
                    "author": parsed_data.get("author_name", ""),
                    "category": parsed_data.get("category_name", ""),
                    "price": parsed_data.get("price", ""),
                    "image": parsed_data.get("image", "")
                })
    elif hasattr(results, "ids"):
        for doc_id in results.ids:
            doc = index.docstore.docs.get(doc_id)
            if doc:
                parsed_data = doc.metadata
                if parsed_data.get("id") not in recommend_books:
                    recommended_books.append({
                        "id": parsed_data.get("id", ""),
                        "title": parsed_data.get("title", ""),
                        "author": parsed_data.get("author_name", ""),
                        "category": parsed_data.get("category_name", ""),
                        "price": parsed_data.get("price", ""),
                        "image": parsed_data.get("image", "")
                    })

    return {"recommended_books": recommended_books}

# === MAIN ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ChatBotLLmaIndex:app", host="127.0.0.1", port=8000, reload=False)
