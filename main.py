import subprocess
import threading
from src.utils.env_loader import load_env_vars

env_vars = load_env_vars()

def run_backend():
    subprocess.run(["uvicorn", "src.backend.api.main:app", "--reload", "--port", f"{env_vars['API_PORT']}"])

def run_frontend():
    subprocess.run(["python", "-m", "src.frontend.main"])

if __name__ == "__main__":
    t1 = threading.Thread(target=run_backend)
    t1.start()

    run_frontend()
