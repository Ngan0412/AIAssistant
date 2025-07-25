from fastapi import FastAPI
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex
from llama_index.readers.database import DatabaseReader
from llama_index.llms.openrouter import OpenRouter
from sqlalchemy import create_engine
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # ✅ dùng cho LangChain với OpenRouter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from llama_index.core.retrievers import BaseRetriever as LlamaRetriever
from pydantic import Field
# from llama_index.core.schema import TextNode
# from sqlalchemy import text
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
import os
from cryptography.fernet import Fernet
import base64, hashlib, getpass

# === FASTAPI ===
app = FastAPI()

# === INPUT ===
class QueryRequest(BaseModel):
    question: str

# === INIT API KEY ===
encrypted_api_key = "gAAAAABog0FSo-Xam59eQZgtSuFnOmEDi4RLoZwvLX4bbvicn0rVM5814RtRNlTOZskoXuotOGv5eOjxrBglt6qtWu2wFzIQZImosqWm83vjdWul4szncWushwiZs01OMv9GWR-c_O9xx503jFDaOrgEXaE-Rr9wYigVwgqB73jOJpOugC8DM5U="
OPENROUTER_API_KEY = None
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
qa_chain = None  # <-- để toàn cục dùng được
def check_pass():
    global OPENROUTER_API_KEY
    print("🔑 Nhập mật khẩu để giải mã API Key:")
    password = getpass.getpass()
    key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
    f = Fernet(key)
    OPENROUTER_API_KEY = f.decrypt(encrypted_api_key.encode()).decode()
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# === DB INIT ===
def init_qa_chain():
    global qa_chain, OPENROUTER_API_KEY
    if OPENROUTER_API_KEY is None:
        check_pass()
    connection_string = (
    "mssql+pyodbc://@localhost/BOOK_SHOP_API?"
    "driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    )
    engine = create_engine(connection_string)

    # --- ĐỌC DỮ LIỆU ---
    # with engine.connect() as conn:
    #     result = conn.execute(text("SELECT * FROM book"))
    #     rows = result.fetchall()
    # print("Số dòng truy vấn được:", len(rows))
    # documents = [
    #     TextNode(text=f"title: {row.title}, price: {row.price}")
    #     for row in rows
    # ]
    reader = DatabaseReader(engine=engine)
    documents = reader.load_data(query= """
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

        def _get_relevant_documents(self, query: str) -> List[Document]:
            results = self.llama_retriever.retrieve(query)
            return [
                Document(page_content=node.get_content(), metadata=node.metadata)
                for node in results
            ]
    llama_retriever = index.as_retriever(similarity_top_k=1)
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

# === GỌI CHAT ===
@app.post("/chat")
def ask_bookshop(request: QueryRequest):
    init_qa_chain()
    if qa_chain is None:
        return {"error": "Chưa khởi tạo mô hình, hãy chạy lại ứng dụng."}
    result = qa_chain.invoke({"query": request.question})
    return {"answer": result['result']}


# === MAIN ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ChatBotLLmaIndex:app", host="127.0.0.1", port=8000, reload=False)







# from fastapi import FastAPI
# from pydantic import BaseModel
# from llama_index.core import VectorStoreIndex
# from llama_index.readers.database import DatabaseReader
# from llama_index.llms.openrouter import OpenRouter
# from sqlalchemy import create_engine
# from langchain.chains import RetrievalQA
# from langchain_openai import ChatOpenAI  # ✅ dùng cho LangChain với OpenRouter
# from langchain_core.retrievers import BaseRetriever
# from langchain_core.documents import Document
# from typing import List
# from llama_index.core.retrievers import BaseRetriever as LlamaRetriever
# from pydantic import Field
# # from llama_index.core.schema import TextNode
# # from sqlalchemy import text
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


# # --- CẤU HÌNH GPT-4o TỪ OpenRouter ---
# OPENROUTER_API_KEY = ""  # ✅ Thay bằng API key thực của bạn
# OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# # FastAPI instance
# app = FastAPI(title="Book Query API", description="Dùng LlamaIndex + LangChain + GPT-4o qua OpenRouter")

# --- KẾT NỐI DATABASE ---
# connection_string = (
#     "mssql+pyodbc://@localhost/BOOK_SHOP_API?"
#     "driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
# )
# engine = create_engine(connection_string)

# # --- ĐỌC DỮ LIỆU ---
# # with engine.connect() as conn:
# #     result = conn.execute(text("SELECT * FROM book"))
# #     rows = result.fetchall()
# # print("Số dòng truy vấn được:", len(rows))
# # documents = [
# #     TextNode(text=f"title: {row.title}, price: {row.price}")
# #     for row in rows
# # ]
# reader = DatabaseReader(engine=engine)
# documents = reader.load_data(query= """
# SELECT 
#     book.id,
#     book.title,
#     book.image,
#     book.quantity,
#     book.price,
#     author.name AS author_name,
#     category.name AS category_name,
#     publisher.name AS publisher_name
# FROM 
#     book
# JOIN author ON book.authorid = author.id
# JOIN category ON book.categoryid = category.id
# JOIN publisher ON book.publisherid = publisher.id
# WHERE 
#     book.isdeleted = 0;
# """)
# # --- TẠO LLM CHO LLamaIndex ---
# llama_llm =  OpenRouter(
#     api_key=OPENROUTER_API_KEY,
#     max_tokens=100,
#     context_window=4096,
#     model="openai/gpt-4o",
# )

# # --- TẠO INDEX ---
# index = VectorStoreIndex.from_documents(
#     documents,
#     llm=llama_llm,
#     llm_predictor=llama_llm,
#     embed_model=embed_model
# )

# class CustomRetriever(BaseRetriever):
#     llama_retriever: LlamaRetriever = Field()  # ✅ Khai báo đúng cách

#     def _get_relevant_documents(self, query: str) -> List[Document]:
#         results = self.llama_retriever.retrieve(query)
#         return [
#             Document(page_content=node.get_content(), metadata=node.metadata)
#             for node in results
#         ]
# llama_retriever = index.as_retriever(similarity_top_k=5)
# langchain_retriever = CustomRetriever(llama_retriever=llama_retriever)
# # --- LangChain LLM ---
# langchain_llm = ChatOpenAI(
#     base_url=OPENROUTER_API_BASE,
#     api_key=OPENROUTER_API_KEY,
#     model="openai/gpt-4o",
#     max_tokens=100
# )

# # --- KẾT NỐI VỚI LANGCHAIN ---
# qa_chain = RetrievalQA.from_chain_type(
#     llm=langchain_llm,
#     retriever=langchain_retriever,
#     return_source_documents=True
# )

# # --- MODEL INPUT CHO API ---
# class QueryRequest(BaseModel):
#     question: str

# # --- ROUTE ---
# @app.post("/chat")
# def ask_bookshop(request: QueryRequest):
#     result = qa_chain.invoke({"query": request.question})
#     return {"answer": result['result']}
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("ChatBotLLmaIndex:app", host="127.0.0.1", port=8000, reload=True)