import boto3

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='danh',
    aws_secret_access_key='danh2606'
)
s3.download_file('knowledge-base', 'Trung Nguyen Legend.xlsx', 'trungnguyen.xlsx')
