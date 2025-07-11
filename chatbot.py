import os
import uuid
import streamlit as st
import google.generativeai as gpt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres.vectorstores import PGVector
from history.session import get_session_id
from history.history import ChatHistory
from rewrite.rewrite import Rewriter
from embedding.embedding import get_question, STEmbeddings
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"] = "AIzaSyAyUApuUBGuFdYTIlZ8FWZGbJrw6RHqUqU"
os.environ["STREAMLIT_WATCH_MODULES"] = "false"

def map_role(role):
    if role == "model":
        return "assistant"
    else:
        return role

st.set_page_config(
    page_title="Trung Nguyên ChatBot",
    page_icon="☕",
    layout="wide",
)

gpt.configure(api_key=api_key)
model = gpt.GenerativeModel('gemini-1.5-pro')

llm_answer = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
llm_rewrite = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

prompt = ChatPromptTemplate.from_template(
"""
Trả lời chỉ dựa trên tài liệu được cung cấp.
Hãy đưa ra câu trả lời chính xác nhất được cung cấp trong tài liệu.
Nếu câu hỏi trùng khớp với câu hỏi mẫu trong tài liệu thì đưa ra trả lời y chang với câu trả lời tương ứng trong tài liêu, không thay đổi hay thêm bớt
<context>
{context}
<context>
Questions:{input}
"""
)
embeddings = STEmbeddings()
connection = 'postgresql+psycopg://danh:danh2606@localhost:5432/trungnguyen'
collection_name = "trungnguyen_docs"
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
sample = get_question()

if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {}
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

if st.sidebar.button("➕ Tạo phiên chat mới"):
    new_session_id = str(uuid.uuid4())
    st.session_state.all_sessions[new_session_id] = {
        "chat_history": ChatHistory(new_session_id),
        "chat_session": []
    }
    st.session_state.active_session_id = new_session_id

if st.session_state.all_sessions:
    selected = st.sidebar.selectbox("Chọn phiên chat:", list(st.session_state.all_sessions.keys()), index=0)
    st.session_state.active_session_id = selected

if st.session_state.active_session_id:
    session = st.session_state.all_sessions[st.session_state.active_session_id]
    chat_history = session["chat_history"]
    chat_session = session["chat_session"]

    st.title("☕ Trung Nguyên ChatBot")

    for msg in chat_session:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Nhập câu hỏi của bạn")
    if query: 
        st.chat_message("user").markdown(query)
        chat_session.append({"role": "user", "content": query})
        chat_history.add_user_message(query)
        full_hist = chat_history.get_formatted()
        rewriter = Rewriter(llm_rewrite)
        standalone = rewriter.rewrite(question=query, history=full_hist, sample=sample)
        st.chat_message("assistant").markdown(f"Câu hỏi đầy đủ bạn là: {standalone}")
        chat_session.append({"role": "assistant", "content": f"Câu hỏi đầy đủ bạn là: {standalone}"})

        document_chain = create_stuff_documents_chain(llm_answer,prompt)
        retriever = vector_store.as_retriever(search_kwargs={"k": 30})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input':standalone})
        chat_history.add_assistant_message(response['answer'])

        st.chat_message("assistant").markdown(response['answer'])
        chat_session.append({"role": "assistant", "content": response['answer']})

        with open("history\history.txt", "w", encoding="utf-8") as f:
            for message in chat_history.get_messages():
                f.write(f"{message}\n")
        print(len(response["context"]))

        for i, doc in enumerate(response["context"]):
            print(doc.metadata)

else:
    st.title("☕ Trung Nguyên ChatBot")
    st.info("Bấm nút 'Tạo phiên chat mới' để bắt đầu.")


