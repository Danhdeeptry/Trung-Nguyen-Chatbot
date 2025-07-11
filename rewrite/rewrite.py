from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage

# a tiny prompt to rewrite
_REWRITE_TMPL = """
Bạn là một trợ lý có nhiệm vụ viết lại câu hỏi của người dùng sao cho đầy đủ ý nghĩa và ngữ cảnh dựa trên lịch sử đoạn hội thoại.
Đây là lịch sử đoạn hội thoại:
{history}

Đây là danh sách câu hỏi mẫu:
{sample}

Đây là câu hỏi mới của người dùng:
"{question}"

Nếu câu hỏi chứa các đại từ thay thế mơ hồ như "nó", "họ", "anh ấy", "chị ấy", "cái đó" hãy dựa vào lịch sử hội thoại và danh sách câu hỏi mẫu để thay thế bằng từ/cụm từ cụ thể, đầy đủ ý nghĩa.
Nếu câu hỏi trùng khớp với câu hỏi trong danh sách câu hỏi mẫu thì giữ nguyên câu hỏi gốc, không cần giải thích gì thêm.
Nếu không tìm thấy thông tin hoặc ngữ cảnh phù hợp trong lịch sử đoạn hội thoại và danh sách câu hỏi mẫu để viết lại thì hãy giữ nguyên câu hỏi gốc, không cần giải thích gì thêm.
Nếu câu hỏi đã rõ ràng và đầy đủ ý nghĩa thì cũng giữ nguyên câu hỏi gốc.
Chỉ trả về câu hỏi sau khi đã viết lại hoặc giữ nguyên, không cần giải thích thêm.
"""
class Rewriter:
    def __init__(self, llm):
        prompt = ChatPromptTemplate.from_template(_REWRITE_TMPL)
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def rewrite(self, question: str, history: str, sample: str) -> str:
        out = self.chain.run({"history": history, "sample": sample, "question": question})
        return out.strip()