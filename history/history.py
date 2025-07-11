from langchain_core.chat_history import InMemoryChatMessageHistory

class ChatHistory:
    def __init__(self, session_id: str, max_turns: int = 20):
        self._hist = InMemoryChatMessageHistory()
        self.max_messages = max_turns * 2 

    def add_user_message(self, text: str):
        self._hist.add_user_message(text)
        self._trim_history()

    def add_assistant_message(self, text: str):
        self._hist.add_ai_message(text)
        self._trim_history()

    def _trim_history(self):
        if len(self._hist.messages) > self.max_messages:
            self._hist.messages = self._hist.messages[-self.max_messages:]

    def get_messages(self):
        return self._hist.messages

    def get_formatted(self):
        out = []
        for msg in self._hist.messages:
            role = "User" if msg.type == "human" else "Assistant"
            out.append(f"{role}: {msg.content}")
        return "\n".join(out)