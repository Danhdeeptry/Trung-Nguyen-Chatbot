import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base
from utils.env_loader import get_db_connection_string

DATABASE_URI = get_db_connection_string()

engine = create_engine(DATABASE_URI)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)