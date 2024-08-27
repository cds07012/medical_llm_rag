import os
import boto3
import fitz  # PyMuPDF
from tqdm import tqdm
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# PDF 파일을 로컬에서 사용하거나 S3에서 다운로드하는 함수
def get_pdfs(bucket_name, prefix, download_path='/tmp/pdfs'):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    pdf_files = []
    total_files = len(response.get('Contents', []))
    
    for obj in tqdm(response.get('Contents', []), desc="Checking PDFs"):
        if obj['Key'].endswith('.pdf'):
            file_name = os.path.join(download_path, obj['Key'].split('/')[-1])
            
            # 로컬에 파일이 있으면 다운로드 생략
            if os.path.exists(file_name):
                print(f"Using existing local file: {file_name}")
            else:
                print(f"Downloading {file_name} from S3...")
                s3_client.download_file(bucket_name, obj['Key'], file_name)
            
            pdf_files.append(file_name)
    
    return pdf_files

# S3에서 벡터 DB 파일을 다운로드하는 함수
def download_vector_db_from_s3(bucket_name, vector_db_s3_path, local_path='/tmp/vectordb_index'):
    s3_client = boto3.client('s3')
    required_files = ['index.faiss', 'index.pkl']
    
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    for file in required_files:
        s3_file_path = os.path.join(vector_db_s3_path, file)
        local_file_path = os.path.join(local_path, file)
        try:
            s3_client.download_file(bucket_name, s3_file_path, local_file_path)
            print(f"Downloaded {file} from S3 to {local_file_path}")
        except s3_client.exceptions.NoSuchKey:
            print(f"{s3_file_path} not found in S3.")
            return False  # 파일 중 하나라도 없으면 False 반환
    
    return True  # 모든 파일이 성공적으로 다운로드되면 True 반환

# 벡터 DB가 존재하는지 확인하는 함수
def check_existing_vector_db(local_path, bucket_name, vector_db_s3_path):
    required_files = ['index.faiss', 'index.pkl']
    
    # 로컬에서 확인
    if all(os.path.exists(os.path.join(local_path, file)) for file in required_files):
        return True
    
    # 로컬에 없으면 S3에서 확인 및 다운로드 시도
    return download_vector_db_from_s3(bucket_name, vector_db_s3_path, local_path)

# PyMuPDF를 사용하여 PDF에서 텍스트 추출 및 메타데이터 수집
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        title = doc.metadata.get('title', os.path.basename(pdf_path))  # 파일 이름을 기본 제목으로 설정
        documents = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # 비어있는 텍스트는 무시
                metadata = {
                    'title': title,
                    'page_number': page_num + 1
                }
                documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

# Bedrock 임베딩을 사용하여 벡터 DB 생성 및 기존 DB에 추가
def create_and_append_bedrock_index(pdf_paths, bucket_name, vector_db_s3_path='vectorDB_append/'):
    s3_client = boto3.client('s3')

    # Bedrock 임베딩 사용
    embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        region_name='us-east-1',
        model_id='amazon.titan-embed-text-v1'
    )

    # 기존 벡터 DB 로드 시도
    local_index_path = '/tmp/vectordb_index'
    if check_existing_vector_db(local_index_path, bucket_name, vector_db_s3_path):
        vector_index = FAISS.load_local(local_index_path, embeddings, allow_dangerous_deserialization=True)
        print("Existing vector DB loaded.")
    else:
        vector_index = None
        print("No existing vector DB found, creating a new one.")

    # 문서 로드 및 텍스트 추출
    pdf_count = 0
    for path in tqdm(pdf_paths, desc="Processing PDFs"):
        try:
            documents = extract_text_from_pdf(path)
            
            if not documents:
                print(f"Skipping {path} due to empty content.")
                continue
            
            # 벡터 DB가 존재하면 새 문서 추가, 아니면 새로 생성
            if vector_index:
                vector_index.add_documents(documents)
            else:
                vector_index = FAISS.from_documents(documents, embeddings)

            pdf_count += 1

            # 10개의 PDF마다 벡터 DB 저장
            if pdf_count % 10 == 0 or pdf_count == len(pdf_paths):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                current_index_path = f'/tmp/vectordb_index_{timestamp}'
                os.makedirs(current_index_path, exist_ok=True)
                
                print(f"Saving vector DB to local at {current_index_path}...")
                vector_index.save_local(current_index_path)
                
                print(f"Uploading vector DB to S3...")
                for file_name in tqdm(os.listdir(current_index_path), desc="Uploading vector DB to S3"):
                    s3_client.upload_file(
                        os.path.join(current_index_path, file_name), 
                        bucket_name, 
                        os.path.join(vector_db_s3_path, file_name)
                    )
                
                print(f"Vector DB files uploaded to s3://{bucket_name}/{vector_db_s3_path}")
        
        except Exception as e:
            print(f"Error processing {path}: {e}. Skipping to next file.")

if __name__ == "__main__":
    bucket_name = 'snuh-data-team2'
    prefix = 'data/'
    
    # 로컬에 파일이 있으면 사용하고, 없으면 S3에서 다운로드
    pdf_paths = get_pdfs(bucket_name, prefix)
    print("PDFs ready for processing.")
    
    # 벡터 DB 생성 및 S3에 저장
    create_and_append_bedrock_index(pdf_paths, bucket_name)
