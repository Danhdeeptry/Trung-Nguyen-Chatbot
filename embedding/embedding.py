import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def embed_text(text: str) -> np.ndarray:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=400, chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    embs = model.encode(chunks, convert_to_numpy=True)
    return np.mean(embs, axis=0)

class STEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [embed_text(t).tolist() for t in texts]
    def embed_query(self, text):
        return embed_text(text).tolist()
    
def get_question():
    list_question = "\n".join(doc.metadata.get("question", "") for doc in documents)
    return list_question
    
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
file_path = os.path.join(BASE_DIR, 'load_data', 'trungnguyen.xlsx')

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
embeddings = STEmbeddings()

connection = 'postgresql+psycopg://danh:danh2606@localhost:5432/trungnguyen'
collection_name = "trungnguyen_docs"
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
vector_store.create_collection()       
vector_store.delete_collection()       
vector_store.create_collection()
vector_store.add_documents(documents=documents)

