import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import boto3
from utils.env_loader import load_env_vars
env_vars = load_env_vars()

def download_file(bucket: str, file: str, local_path):
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=env_vars["MINIO_ENDPOINT"],
            aws_access_key_id=env_vars["MINIO_ACCESS_KEY"],
            aws_secret_access_key=env_vars["MINIO_SECRET_KEY"]
        )
        s3.download_file(bucket, file, local_path)
        print(f"✅ Download complete: {local_path}")
    except Exception as e:
        print(f"❌ Failed to download file from S3: {e}")
