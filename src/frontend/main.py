import os
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from utils.env_loader import load_env_vars

env_vars = load_env_vars()
def main():
    # Get path to this file's directory
    src_dir = Path(__file__).resolve().parent

    # Path to the Streamlit frontend file
    app_path = src_dir / "app.py"

    if not app_path.exists():
        print(f"❌ Cannot find app.py at {app_path}")
        sys.exit(1)

    # Run Streamlit
    try:
        print("🚀 Starting Trung Nguyên ChatBot UI...")
        subprocess.run(["streamlit", "run", str(app_path), "--server.port", f"{env_vars['STREAMLIT_PORT']}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()