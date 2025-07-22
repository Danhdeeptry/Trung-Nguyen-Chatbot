from db.connection import SessionLocal
from db.models import ChatSession, Message

class ChatHistory:
    def __init__(self, session_id: str, max_turns: int = 20):
        self.session_id = session_id
        self.max_messages = max_turns * 2
        self.db = SessionLocal()
        self._ensure_session_exists()

    def _ensure_session_exists(self):
        session = self.db.query(ChatSession).filter_by(session_id=self.session_id).first()
        if not session:
            session = ChatSession(session_id=self.session_id)
            self.db.add(session)
            self.db.commit()

    def add_user_message(self, text: str):
        self._add_message("user", text)

    def add_assistant_message(self, text: str):
        self._add_message("assistant", text)

    def _add_message(self, role: str, text: str):
        session = self.db.query(ChatSession).filter_by(session_id=self.session_id).first()
        msg = Message(session_id=session.id, role=role, content=text)
        self.db.add(msg)
        self.db.commit()
        # self._trim_history(session)

    # def _trim_history(self, session):
    #     messages = self.db.query(Message).filter_by(session_id=session.id).order_by(Message.timestamp).all()
    #     if len(messages) > self.max_messages:
    #         to_delete = messages[:len(messages) - self.max_messages]
    #         for msg in to_delete:
    #             self.db.delete(msg)
    #         self.db.commit()

    def get_messages(self):
        session = self.db.query(ChatSession).filter_by(session_id=self.session_id).first()
        return self.db.query(Message).filter_by(session_id=session.id).order_by(Message.timestamp).all()

    def get_formatted(self):
        messages = self.get_messages()
        out = []
        for msg in messages:
            role = "User" if msg.role == "human" else "Assistant"
            out.append(f"{role}: {msg.content}")
        return "\n".join(out)
