import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

def prepare_chroma_db_all(data_dir: str, persist_dir: str):
    """
    전체 텍스트 문서를 하나의 벡터 DB로 통합 저장
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_documents = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue
            file_path = os.path.join(root, file)
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                text = loader.load()[0].page_content
                doc = Document(page_content=text, metadata={"source": file_path})
                all_documents.append(doc)
                print(f"✅ 로드 완료: {file_path}")
            except Exception as e:
                print(f"❌ 에러: {file_path} - {str(e)}")

    # 통합 벡터 DB 저장
    vector_db = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print(f"🎉 통합 저장 완료! 총 {len(all_documents)}개 문서")

if __name__ == "__main__":
    data_dir = "./data"               # 예시: 'data/' 디렉토리 전체 탐색
    persist_dir = "./vectorDB"    # 하나의 DB로 저장
    prepare_chroma_db_all(data_dir, persist_dir)
