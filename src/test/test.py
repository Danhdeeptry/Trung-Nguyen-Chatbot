import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer
# from langchain.embeddings.base import Embeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils.env_loader import load_env_vars
from embedding.embedding import Embedding


env_vars = load_env_vars()
api_key = env_vars['GOOGLE_API_KEY']

with open("test\\testcases.json", "r", encoding="utf-8") as f:
    testcases = json.load(f)
llm_answer = ChatGoogleGenerativeAI(model=env_vars['LLM_MODEL'], temperature=0.7)
llm_eval = ChatGoogleGenerativeAI(model=env_vars['LLM_MODEL'], temperature=0)
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

prompt = ChatPromptTemplate.from_template(
"""
Trả lời chỉ dựa trên tài liệu được cung cấp.
Hãy đưa ra câu trả lời chính xác nhất được cung cấp trong tài liệu.
Nếu câu hỏi trùng khớp với câu hỏi mẫu trong tài liệu thì đưa ra trả lời y chang với câu trả lời tương ứng trong tài liêu, không thay đổi hay thêm bớt
Câu trả lời phải giữ nguyên toàn bộ câu chữ trong tài liệu, trả lời đầy đủ thông tin được cung cấp và cách dòng ở những chỗ hợp lí
<context>
{context}
<context>
Questions:{input}
"""
)

compare_prompt = """
Bạn là một chuyên gia đánh giá chatbot.

Dưới đây là hai câu trả lời cho cùng một câu hỏi:

Câu hỏi: {question}

Câu trả lời mẫu:
{expected_answer}

Câu trả lời chatbot:
{actual_answer}

Nhiệm vụ:
Cho biết liệu câu trả lời của chatbot có đúng và đầy đủ thông tin giống câu trả lời mẫu** hay không (Chỉ trả lời: Yes / No).

"""

# def embed_text(text: str) -> np.ndarray:
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         tokenizer, chunk_size=400, chunk_overlap=100
#     )
#     chunks = splitter.split_text(text)
#     embs = model.encode(chunks, convert_to_numpy=True)
#     return np.mean(embs, axis=0)

# class STEmbeddings(Embeddings):
#     def embed_documents(self, texts):
#         return [embed_text(t).tolist() for t in texts]
#     def embed_query(self, text):
#         return embed_text(text).tolist()
    
def evaluate_with_llm(question, expected_answer, actual_answer):
    filled_prompt = compare_prompt.format(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer
    )
    response = llm_eval.invoke(filled_prompt)
    return response.content.strip()
    
# gc = gspread.service_account(filename="gen-lang-client-0373313638-f4d04baa9526.json")
# sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1K9YoET7e_JGO4yG0bvsghEHdW17rcUkxJzKoIpdI9ME/edit?gid=0#gid=0")
# ws = sheet.worksheet("Sheet1")
# data = pd.DataFrame(ws.get_all_records())
# data = data[["Câu hỏi", "Câu trả lời"]].rename(
#     columns={"Câu hỏi":"question", "Câu trả lời":"answer"}
# )
# docs = []
# for i, row in data.iterrows():
#     text = " | ".join(f"{col}: {row[col]}" for col in data.columns)
#     content = f"Câu hỏi: {row['question']}\nCâu trả lời: {row['answer']}"
#     docs.append(Document(page_content=content, metadata={"index": i,"question": row['question']}))
embedding = Embedding()
sample = embedding.get_question()
vector_store = embedding._init_vector_store()
# vectors = [embed_text(d.page_content) for d in docs]
# dim = vectors[0].shape[0]e
# index = faiss.IndexFlatL2(dim)
# index.add(np.vstack(vectors))
# store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=InMemoryDocstore({str(i): d for i,d in enumerate(docs)}),
#     index_to_docstore_id={i: str(i) for i in range(len(docs))},
# )

def evaluate_testcase(tc):
    print(f"Running: {tc['id']}")
    query = tc['query']
    actual_answer = tc['answer']
    expected_metadata_index = tc['metadata_index']

    # Retrieval + Answer
    document_chain = create_stuff_documents_chain(llm_answer, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 30})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': query})
    expected_answer = response['answer']
    context_docs = response['context']

    # Check metadata chunk match
    answer_pass = evaluate_with_llm(query, expected_answer, actual_answer) == "Yes"
    metadata_pass = any(expected_metadata_index == doc.metadata.get('index') for doc in context_docs)

    result = {
        "id": tc['id'],
        "query": query,
        "answer": expected_answer,
        "answer_pass": answer_pass,
        "metadata_pass": metadata_pass,
        "all_pass": answer_pass and metadata_pass
    }

    return result

results = []
for tc in testcases:
    result = evaluate_testcase(tc)
    results.append(result)
    print(result)
    print("=" * 60)
with open("tests.txt", "w", encoding="utf-8") as f:
    f.write(f"{results}")
total = len(results)
passed = sum(1 for r in results if r['all_pass'])
print(f"\nPassed {passed}/{total} testcases ({passed/total*100:.1f}%)")


