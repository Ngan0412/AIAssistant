from langchain_ollama import OllamaLLM
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# 1. Kết nối MySQL
db = SQLDatabase.from_uri("mysql+pymysql://root:123456@localhost:3306/base")

# 2. Dùng LLM qua Ollama (Allama local model)
llm = OllamaLLM(model="gemma3:4b")  

# 3. Tạo SQL Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# 4. Hỏi câu tiếng Việt
question = "Có bao nhiêu sản phẩm thuộc thể loại nước hoa"
result = agent_executor.invoke({"input": question})

print(result)
