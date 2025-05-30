import os
import requests
import xml.etree.ElementTree as ET
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

def fetch_api_docs(service_key: str, max_page: int = 1) -> list[Document]:
    """
    여성가족부 아이돌보미 보수교육 API 호출 → XML 파싱 → LangChain Document로 변환
    """
    docs = []
    for page in range(1, max_page + 1):
        url = (
            "http://apis.data.go.kr/1383000/idis/refresherEducationService/getRefresherEducationList"
            f"?serviceKey={service_key}"
            f"&pageNo={page}&numOfRows=10&type=xml"
        )

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            for item in root.iter("item"):
                crtrYr = item.findtext("crtrYr", default="N/A")
                eduDvsnNm = item.findtext("eduDvsnNm", default="N/A")
                eduCrsNm = item.findtext("eduCrsNm", default="N/A")
                eduCn = item.findtext("eduCn", default="N/A")
                date = item.findtext("dataCrtrYmd", default="N/A")

                text = (
                    f"[{date}] 기준연도: {crtrYr}, 교육구분: {eduDvsnNm}, "
                    f"과정명: {eduCrsNm}, 내용: {eduCn}"
                )
                docs.append(Document(page_content=text, metadata={"source": f"RefresherEdu_{date}"}))

        except Exception as e:
            print(f"❌ API 페이지 {page} 요청 실패: {e}")

    print(f"🌐 보수교육 API 문서 {len(docs)}개 로드 완료")
    return docs

def prepare_chroma_db_all(data_dir: str, persist_dir: str, service_key: str):
    """
    텍스트 파일 + API XML 데이터 → 벡터 DB 저장
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_documents = []

    # 1. 텍스트 문서 로드
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
                print(f"✅ 파일 로드 완료: {file_path}")
            except Exception as e:
                print(f"❌ 파일 에러: {file_path} - {str(e)}")

    # 2. API XML 로드
    api_docs = fetch_api_docs(service_key)
    all_documents.extend(api_docs)

    # 3. Chroma DB 저장
    vector_db = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print(f"🎉 총 {len(all_documents)}개 문서 벡터 저장 완료!")
    
    # 저장된 벡터 DB 불러오기
    vector_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )

    query = "아동 심리 교육은 어떤 내용을 포함하나요?"
    docs = vector_db.similarity_search(query, k=3)

    for i, doc in enumerate(docs, 1):
        print(f"\n🔍 [{i}위 결과]")
        print(doc.page_content)

# ───────────────────────────────
# 🧪 실행 예시
if __name__ == "__main__":
    data_dir = "./data"
    persist_dir = "./vectorDB"

    # 📌 Encoding된 서비스 키 사용!
    service_key = "gGxYEcfSjjqDpdu6Amo9r9uY%2FPL9yiXpWYpu1%2BPGbk9sKfjAdwP%2FjXzfI0d%2BEB%2B80Ew4Wd4AB%2FiMd1hjpWy9SA%3D%3D"

    prepare_chroma_db_all(data_dir, persist_dir, service_key)
