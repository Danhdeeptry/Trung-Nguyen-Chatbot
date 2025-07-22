import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from history.history import ChatHistory
from rewrite.rewrite import Rewriter
from embedding.embedding import Embedding
from backend.logger import setup_logger
from backend.config import llm_answer, llm_rewrite, vector_store, prompt
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from requests.exceptions import HTTPError, RequestException
from google.api_core import exceptions as google_exceptions
from fastapi import HTTPException

logger = setup_logger()
embedding = Embedding()
list_question = embedding.get_question()

def run_chat_session(query, chat_history: ChatHistory, chat_session: str, list_question: str = list_question):
    logger.info(f"[Session ID: {chat_session}] - User query: {query}")

    try:
        rewriter = Rewriter(llm_rewrite)
        standalone = rewriter.rewrite(question=query, history=chat_history.get_formatted(), sample=list_question)
        logger.info(f"[Session ID: {chat_session}] - User query: {query} - Rewritten question: {standalone}")
    except HTTPError as e:
        logger.error(
            "Rewrite HTTP Error - "
            f"Status: {e.response.status_code} - "
            f"Response: {e.response.text} - "
            f"URL: {e.request.url}"
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e

    except google_exceptions.GoogleAPIError as e:
        if isinstance(e, google_exceptions.Unauthenticated):
            logger.error(
                "Gemini API Authentication Failed - "
                f"Error: {str(e)} - "
                "Verify GOOGLE_API_KEY is valid and has Gemini permissions"
            )
            raise HTTPException(status_code=500, detail="Internal Server Error") from e
        elif isinstance(e, google_exceptions.PermissionDenied):
            logger.error(
                "Gemini API Permission Denied - "
                f"Error: {str(e)} - "
                "Check if Gemini API is enabled in Google Cloud"
            )
            raise HTTPException(status_code=500, detail="Internal Server Error") from e
        else:
            logger.error(
                "Google API Error - "
                f"Type: {type(e).__name__} - "
                f"Error: {str(e)} - "
                f"HTTP Code: {getattr(e, 'code', 'N/A')}"
            )
            raise HTTPException(status_code=500, detail="Internal Server Error") from e

    except google_exceptions.ResourceExhausted as e:
        logger.error(
            "Quota exceeded - "
            f"Error: {str(e)} - "
        )
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    except RequestException as e:
        logger.error(
            "Rewrite Network Error - "
            f"Type: {type(e).__name__} - "
            f"Message: {str(e)} - "
            f"URL: {getattr(e.request, 'url', 'N/A')}"
        )
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    except Exception as e:
        logger.error(
            "Rewrite Unexpected Error - "
            f"Type: {type(e).__name__} - "
            f"Error: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")

    document_chain = create_stuff_documents_chain(llm_answer, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 30})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    try:
        response = retrieval_chain.invoke({'input': standalone})
    except HTTPError as e:
        logger.error(
            "Response HTTP Error - "
            f"Status: {e.response.status_code} - "
            f"Response: {e.response.text} - "
            f"URL: {e.request.url}"
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e

    except google_exceptions.ResourceExhausted as e:
        logger.error(
            "Quota exceeded - "
            f"Error: {str(e)} - "
        )
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    except RequestException as e:
        logger.error(
            "Response Network Error - "
            f"Type: {type(e).__name__} - "
            f"Message: {str(e)} - "
            f"URL: {getattr(e.request, 'url', 'N/A')}"
        )
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    except Exception as e:
        logger.error(
            "Response Unexpected Error - "
            f"Type: {type(e).__name__} - "
            f"Message: {str(e)} - "
            f"Traceback: {logging.traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    chat_history.add_user_message(query)
    chat_history.add_assistant_message(response["answer"])

    chunk_infos = [
        f"[{i}] index: {doc.metadata.get('index', '')}"
        for i, doc in enumerate(response["context"])
    ]
    logger.info(f"[Session ID: {chat_session}] - User query: {query} - Retrieved Chunks: {' || '.join(chunk_infos)}")
    logger.info(f"[Session ID: {chat_session}] - User query: {query} - Response: {response["answer"]}")

    return {
        "standalone": standalone,
        "answer": response["answer"],
        "context": response["context"]
    }