import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

def load_env_vars(env_path: str = None) -> Dict[str, str]:
    if env_path is None:
        env_path = Path(__file__).parent.parent.parent / '.env'
    
    if not load_dotenv(env_path):
        print(f"Warning: No .env file found at {env_path}")
    
    vars = [
        'GOOGLE_API_KEY',
        'LLM_MODEL',
        'EMBEDDING_MODEL',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_DB',
        'POSTGRES_HOST',
        'POSTGRES_PORT',
        'PGVECTOR_COLLECTION',
        'MINIO_ENDPOINT',
        'MINIO_ACCESS_KEY',
        'MINIO_SECRET_KEY',
        'MINIO_BUCKET',
        'MINIO_FILE',
        'STREAMLIT_WATCH_MODULES',
        'API_URL',
        'API_PORT',
        'STREAMLIT_PORT'
    ]
    
    env_vars = {}
    for var in vars:
        value = os.getenv(var)
        if value is None:
            print(f"Warning: Environment variable {var} not found")
        env_vars[var] = value    
    return env_vars

def get_db_connection_string() -> str:
    env_vars = load_env_vars()
    return (
        f"postgresql+psycopg://{env_vars['POSTGRES_USER']}:{env_vars['POSTGRES_PASSWORD']}"
        f"@{env_vars['POSTGRES_HOST']}:{env_vars['POSTGRES_PORT']}/{env_vars['POSTGRES_DB']}"
    ) 