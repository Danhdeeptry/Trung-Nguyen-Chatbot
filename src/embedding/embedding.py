import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import inspect
from utils.env_loader import load_env_vars, get_db_connection_string
from load_data.load_data import download_file
from db.connection import init_db, engine

env_vars = load_env_vars()

class STEmbeddings(Embeddings):
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        embedding_model: str = env_vars['EMBEDDING_MODEL']
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

    def embed_text(self, text: str) -> np.ndarray:
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            model = SentenceTransformer(self.embedding_model)
        except Exception as e:
            print(f"❌ Lỗi khi tải model/tokenizer: {e}")
            raise

        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(text)
        embs = model.encode(chunks, convert_to_numpy=True)
        return np.mean(embs, axis=0)
    
    def embed_documents(self, texts):
        return [self.embed_text(t).tolist() for t in texts]
    
    def embed_query(self, text):
        return self.embed_text(text).tolist()
    
class Embedding:
    def __init__(
        self,
        connection: str = get_db_connection_string(),
        collection_name: str = env_vars['PGVECTOR_COLLECTION'],
        bucket: str = env_vars['MINIO_BUCKET'],
        file_name: str = env_vars['MINIO_FILE'],
        embeddings: Embeddings = STEmbeddings()
    ):
        self.connection = connection 
        self.collection_name = collection_name
        self.bucket = bucket 
        self.file_name = file_name
        self.embeddings = embeddings
        
    
    def _init_vector_store(self):
        try:
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection,
                use_jsonb=True,
            )
        except Exception as e:
            print(f"❌ Lỗi khi kết nối PostgreSQL hoặc khởi tạo PGVector: {e}")
            raise
    
    def get_vector_store(self):
        self._init_vector_store()
        return self.vector_store

    def get_data(self):
        file_path = os.path.join('src', 'load_data', self.file_name)
        if not os.path.exists(file_path):
            download_file(self.bucket, self.file_name, file_path)

        df = pd.read_excel(file_path)
        df = df.fillna('')
        documents = []
        for index, row in df.iterrows():
            question = str(row[df.columns[2]]).strip()
            answer = str(row[df.columns[3]]).strip()
            
            page_content = f"Câu hỏi: {question}\nCâu trả lời: {answer}"
            metadata = {"question": question, "index": index}
            
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        return documents

    def get_question(self):
        list_question = "\n".join(doc.metadata.get("question", "") for doc in self.get_data())
        return list_question
    
    def is_collection_empty(self):
        try:
            docs = self.vector_store.similarity_search("test", k=1)
            return len(docs) == 0
        except Exception as e:
            print(f"⚠️ Could not check collection contents: {e}")
            return True
        
    def ensure_data_stored(self):
        if self.is_collection_empty():
            print("🟡 No data found in vector DB, embedding now...")
            self.store_data()
        else:
            print("✅ Vector DB already contains data, skipping embedding.")
        
    def store_data(self):
        documents = self.get_data()
        try:
            self.vector_store.add_documents(documents=documents)
            print(f"Added {len(documents)} chunks to vector DB successfully")
        except Exception as e:
            print(f"Error adding data to vector DB: {e}")

    def session_initialized(self):
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        required_tables = {"messages", "chat_sessions"}
        if required_tables.issubset(set(existing_tables)):
            print("✅ Sessions table already exists")
        else:
            print("🟡 Sessions table not found. Initializing sessions table...")
            init_db()


