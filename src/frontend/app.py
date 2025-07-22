import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
from backend.logger import setup_logger
from utils.env_loader import load_env_vars
from requests.exceptions import HTTPError


env_vars = load_env_vars()
logger = setup_logger()
api_url = env_vars['API_URL']

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Trung Nguyên ChatBot",
    page_icon="☕",
    layout="wide",
)

# --- Cache session data để giữ qua F5 ---
@st.cache_resource
def get_session_storage():
    return {
        "all_sessions": {},
        "session_titles": {},
        "active_session_id": None
    }

session_data = get_session_storage()

# --- Init session state from cache ---
if "initialized" not in st.session_state:
    st.session_state.all_sessions = session_data["all_sessions"]
    st.session_state.session_titles = session_data["session_titles"]
    st.session_state.active_session_id = session_data["active_session_id"]
    st.session_state.initialized = True

# --- Sidebar: Session Management ---
st.sidebar.title("💬 Lịch sử phiên chat")

def create_new_session():
    try:
        response = requests.get(f"{api_url}/api/sessions/new")
        response.raise_for_status()
        new_session_id = response.json()["session_id"]
        st.session_state.all_sessions[new_session_id] = {"messages": []}
        st.session_state.session_titles[new_session_id] = "Phiên chat mới"
        st.session_state.active_session_id = new_session_id
    except HTTPError as e:
        logger.error(
            "Create New Session HTTP Error - "
            f"Status: {e.response.status_code} - "
            f"Response: {e.response.text} - "
            f"URL: {e.request.url}"
        )
        st.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        logger.error(
            "Create New Session Unexpected Error - "
            f"Type: {type(e).__name__} - "
            f"Error: {str(e)}"
        )
        st.error("Error: Internal Server Error")

st.sidebar.button("➕ Tạo phiên chat mới", on_click=create_new_session)

# Hiển thị danh sách phiên chat
if st.session_state.all_sessions:
    session_id_list = list(st.session_state.all_sessions.keys())[::-1]
    titles = [
        st.session_state.session_titles.get(sid, "Phiên chat mới")
        for sid in session_id_list
    ]

    selected_index = st.sidebar.selectbox(
        "Chọn phiên chat:",
        range(len(session_id_list)),
        format_func=lambda i: titles[i]
    )
    selected_session_id = session_id_list[selected_index]
    st.session_state.active_session_id = selected_session_id

# Hiển thị Session ID
if st.session_state.active_session_id:
    st.sidebar.markdown(f"Session ID: {st.session_state.active_session_id}")

# --- Main Chat UI ---
if st.session_state.active_session_id:
    session_id = st.session_state.active_session_id
    session = st.session_state.all_sessions[session_id]

    st.title("☕ Trung Nguyên ChatBot")

    # Hiển thị các tin nhắn
    for msg in session["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input từ người dùng
    query = st.chat_input("Nhập câu hỏi của bạn")
    if query:
        st.chat_message("user").markdown(query)
        session["messages"].append({"role": "user", "content": query})

        # Đặt tiêu đề nếu còn mặc định
        if st.session_state.session_titles.get(session_id) == "Phiên chat mới":
            st.session_state.session_titles[session_id] = query.strip()

        try:
            response = requests.post(
                f"{api_url}/api/chat",
                json={
                    "query": query,
                    "session_id": session_id
                }
            )
            response.raise_for_status()
            data = response.json()

            # Hiển thị câu hỏi viết lại
            rewritten = f"Câu hỏi đầy đủ của bạn là: {data['standalone']}"
            session["messages"].append({"role": "assistant", "content": rewritten})
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                displayed_text = ""

                for chunk in rewritten:
                    displayed_text += chunk
                    message_placeholder.markdown(displayed_text
                    )
                    time.sleep(0.01)

                message_placeholder.markdown(displayed_text)

            # Hiển thị câu trả lời dần dần
            answer = data["answer"]
            session["messages"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                displayed_text = ""

                for chunk in answer:
                    displayed_text += chunk
                    message_placeholder.markdown(displayed_text)
                    time.sleep(0.01)

                message_placeholder.markdown(displayed_text)

        except HTTPError as e:
            logger.error(
                "Query Processing HTTP Error - "
                f"Status: {e.response.status_code} - "
                f"Response: {e.response.text} - "
                f"URL: {e.request.url}"
            )
            st.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.error(
                "Query Processing Unexpected Error - "
                f"Type: {type(e).__name__} - "
                f"Error: {str(e)}"
            )
            st.error("Error: Internal Server Error")

else:
    st.title("☕ Trung Nguyên ChatBot")
    st.info("Bấm nút 'Tạo phiên chat mới' để bắt đầu.")

# --- Cập nhật lại dữ liệu cache ---
session_data["all_sessions"] = st.session_state.all_sessions
session_data["session_titles"] = st.session_state.session_titles
session_data["active_session_id"] = st.session_state.active_session_id
