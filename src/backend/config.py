import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.env_loader import load_env_vars
from embedding.embedding import Embedding
from backend.logger import setup_logger
from fastapi import HTTPException
from requests.exceptions import HTTPError

env_vars = load_env_vars()
logger = setup_logger()
api_key = env_vars['GOOGLE_API_KEY']

llm_answer = ChatGoogleGenerativeAI(model=env_vars['LLM_MODEL'], temperature=0.7)
llm_rewrite = ChatGoogleGenerativeAI(model=env_vars['LLM_MODEL'], temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
    Trả lời chỉ dựa trên tài liệu được cung cấp.
    Hãy đưa ra câu trả lời chính xác nhất được cung cấp trong tài liệu.
    Nếu câu hỏi trùng khớp với câu hỏi mẫu trong tài liệu thì đưa ra trả lời y chang với câu trả lời tương ứng trong tài liêu, không thay đổi hay thêm bớt, thêm dấu câu và xuống hàng ở những chỗ cần thiết để người dùng dễ hiểu và trình bày dưới dạng markdown.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

embedding = Embedding()
try:
    embedding._init_vector_store()
    embedding.ensure_data_stored()
    embedding.session_initialized()
    vector_store = embedding.get_vector_store()
except HTTPError as e:
    logger.error(
        "DB Connect HTTP Error - "
        f"Status: {e.response.status_code} - "
        f"Response: {e.response.text} - "
        f"URL: {e.request.url}"
    )
    raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
except Exception as e:
    logger.error(
        "DB Connect Unexpected Error - "
        f"Type: {type(e).__name__} - "
        f"Error: {str(e)}"
    )
    raise HTTPException(status_code=500, detail="Internal Server Error")