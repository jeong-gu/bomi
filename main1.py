import pickle
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import openai
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import Base, User, UserRole, Caregiver, UserPreference, Parent
from passlib.context import CryptContext
from fastapi.responses import JSONResponse
import json
import numpy as np
import traceback
import re
from sentence_transformers import SentenceTransformer
import torch

# 환경 변수 로드
load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API 설정
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chat_model = "gpt-4o"

reqBASE_URL = "sqlite:///./users.db"
engine = create_engine(reqBASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



class QueryRequest(BaseModel):
    prompt: str

class ConditionInfo(BaseModel):
    days: List[str]
    times: List[Dict[str, int]]
    special: bool
    age_min: float
    age_max: float
    
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: UserRole
    age: int
    phone: str
    #answers: Optional[list[str]] = None

    conditions: Optional[ConditionInfo] = None
    
    service_type: Optional[int] = None
    hours: Optional[float] = None
    children_count: Optional[int] = None
    income_type: Optional[int] = None
    is_night: Optional[bool] = False
    is_holiday: Optional[bool] = False
    is_multi_child: Optional[int] = 0


class LoginRequest(BaseModel):
    email: str
    password: str

class PreferenceRequest(BaseModel):
    email: str
    summary: str

class ChatPreferenceRequest(BaseModel):
    email: str
    history: list[str]

class RecommendationRequest(BaseModel):
    vectors: Dict[str, List[float]]  # history 대신 vectors를 받도록 수정
    history: List[str]  # 가중치 계산을 위해 필요

class UserChatRequest(BaseModel):
    email: str
    history: List[str]

class CaregiverChatRequest(BaseModel):
    email: str
    history: List[str]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

caregiver_questions = [
    "아이와의 갈등 상황에서, 아이의 마음을 어떻게 이해하려 하시나요?",
    "아이에게 단호하게 훈육해야 할 상황이 온다면, 보통 어떤 방식으로 접근하시나요?",
    "하루 중 아이와 함께하는 루틴이나 중요하게 생각하는 생활 습관이 있으신가요?",
    "아이와 함께하면서 가장 즐거웠던 순간은 언제였나요?",
    "예상치 못한 돌발 상황(예: 아이가 갑자기 아플 때)이 생기면, 어떻게 대처해오셨나요?",
    "아이를 돌봐주시는 분에게 가장 바라는 '마음가짐'은 어떤 모습일까요?",
    "아이와 관계를 맺는 데 있어 중요하다고 느끼는 태도나 자세가 있다면요?",
    "돌봄은 한두 번보다 꾸준함이 중요하다고 하잖아요, 소중하게 생각하는 '꾸준함'이 있다면 어떤 걸까요?",
    "아이를 돌보는 과정에서 가족끼리 나누는 역할이나 분위기는 어떤 편인가요?",
    "누군가에게 중요한 일을 맡기게 될 때, 어떤 부분에서 신뢰를 느끼시나요?",
    "새로운 사람과 처음 만날 때, 어떤 인상을 받을 때 안심이 되시나요?",
    "아이에게 말을 걸거나 대화할 때, 어떤 말투나 표현을 좋아하세요?",
    "부탁한 일이 정확하게 이뤄졌을 때, 어떤 점에서 '잘 해주셨다'고 느끼시나요?",
    "특별히 바쁘거나 도움이 급히 필요할 때, 돌보미가 어떤 태도로 응대해주면 좋을까요?",
]


@app.post("/register")
def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다.")

    hashed_pw = pwd_context.hash(req.password)
    new_user = User(
        username=req.username,
        email=req.email,
        hashed_password=hashed_pw,
        role=req.role,
        age=req.age,
        phone=req.phone
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # ✅ 돌보미 계정이라면 Caregiver row도 생성
    if req.role == UserRole.돌보미:
        caregiver = Caregiver(
            user_id=new_user.id,
            age=req.age,

            # ✅ 벡터 초기값 (모두 0.0)
            parenting_style_vector=json.dumps([0.0] * 8),
            personality_traits_vector=json.dumps([0.0] * 10),
            communication_style_vector=json.dumps([0.0] * 5),
            caregiving_attitude_vector=json.dumps([0.0] * 6),
            handling_situations_vector=json.dumps([0.0] * 4),
            empathy_traits_vector=json.dumps([0.0] * 4),
            trust_time_vector=json.dumps([0.0] * 3)
        )

        # 돌보미 조건이 입력된 경우
        if req.conditions:
            caregiver.available_days = ",".join(req.conditions.days)
            caregiver.available_times = json.dumps(req.conditions.times)
            caregiver.special_child = req.conditions.special
            caregiver.age_min = req.conditions.age_min
            caregiver.age_max = req.conditions.age_max

        db.add(caregiver)
        db.commit()
        logger.info(f"Caregiver row created for user_id={new_user.id}")

    return {
        "message": "회원가입 성공",
        "user_id": new_user.id,
        "role": new_user.role.value
    }


@app.post("/login")
def login_user(req: LoginRequest, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == req.email).first()
        if not user or not pwd_context.verify(req.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 일치하지 않습니다.")
        return {
            "access_token": f"dummy_token_for_{user.email}",
            "username": user.username,
            "role": user.role.value,
            "phone": user.phone,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

logger = logging.getLogger("main")

@app.post("/user/preference")
def save_user_preference(req: PreferenceRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    try:
        # ✅ GPT 임베딩 생성 및 직렬화
        embedding = embedding_model.embed_documents([req.summary])[0]
        pickled_embedding = pickle.dumps(embedding)

        # ✅ 기존 preference 존재 시 update, 없으면 insert
        existing_pref = db.query(UserPreference).filter_by(user_id=user.id).first()

        if existing_pref:
            # 🔄 기존 preference 업데이트
            existing_pref.preferred_style = req.summary
            existing_pref.embedding = pickled_embedding
            logger.info(f"[업데이트] user_id={user.id}")
        else:
            # 🆕 새로운 preference 추가
            new_pref = UserPreference(
                user_id=user.id,
                preferred_style=req.summary,
                embedding=pickled_embedding
            )
            db.add(new_pref)
            logger.info(f"[삽입] user_id={user.id}")

        db.commit()
        return {"message": "성향 저장 완료"}

    except Exception as e:
        db.rollback()
        logger.error(f"성향 분석 실패: {e}")
        raise HTTPException(status_code=500, detail="성향 분석 중 오류 발생")

# SBERT 모델 초기화
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_summary_with_gpt(history: List[str], role: str) -> str:
    """GPT를 사용하여 대화 내용을 요약"""
    prompt = f"""
        당신은 {role}의 성향을 분석하는 전문가입니다.
        아래 대화 내용을 바탕으로 {role}의 성향을 요약해주세요.
        요약은 다음 7개 카테고리에 대한 내용을 포함해야 합니다:
        1. 양육 스타일 (교육, 정서, 자율성, 훈육, 놀이, 안전, 애착, 신체활동)
        2. 성격 특성 (외향성, 내향성, 감성, 이성, 융통성, 원칙성, 꼼꼼함, 자유로움, 유머, 침착함)
        3. 의사소통 스타일 (설명, 직관, 대화, 비언어, 지시)
        4. 돌봄 태도 (인내심, 적극성, 신뢰, 개입, 관찰, 독립)
        5. 상황 대처 (갈등, 돌발상황, 계획, 유연성)
        6. 공감 특성 (감정민감성, 공감, 무던함, 표현)
        7. 신뢰/시간 (시간엄수, 융통성, 신뢰)

        대화 내용:
        {chr(10).join(history)}

        요약:
    """

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": f"당신은 {role}의 성향을 분석하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT 요약 생성 중 오류 발생: {str(e)}")
        return ""

def generate_vectors_from_summary(summary: str) -> Dict[str, List[float]]:
    """SBERT를 사용하여 요약문을 벡터화"""
    try:
        # 각 카테고리별 특성 정의
        categories = {
            "parenting_style_vector": [
                "교육 중심", "정서 케어 중심", "자율성 중심", "훈육 중심",
                "놀이 중심", "안전/보호 중심", "애착 중심", "신체 활동 중심"
            ],
            "personality_traits_vector": [
                "외향적", "내향적", "감성형", "이성형", "융통형", "원칙형",
                "꼼꼼형", "자유형", "유머형", "침착형"
            ],
            "communication_style_vector": [
                "설명 중심", "직관 중심", "대화형", "비언어형", "지시형"
            ],
            "caregiving_attitude_vector": [
                "인내심 있는", "적극적인", "신뢰 중심", "개입형", "관찰형", "독립 유도형"
            ],
            "handling_situations_vector": [
                "갈등 중재형", "돌발 상황 대응형", "계획형", "유연 대응형"
            ],
            "empathy_traits_vector": [
                "감정 민감형", "공감 우선형", "무던한 형", "감정 표현형"
            ],
            "trust_time_vector": [
                "시간 엄수형", "융통성 있는", "신뢰 우선형"
            ]
        }

        # 요약문을 SBERT로 임베딩
        summary_embedding = sbert_model.encode(summary)

        # 각 카테고리별 특성 벡터 생성
        category_embeddings = {}
        for category, traits in categories.items():
            # 각 특성을 SBERT로 임베딩
            trait_embeddings = sbert_model.encode(traits)
            
            # 코사인 유사도 계산
            similarities = []
            for trait_emb in trait_embeddings:
                similarity = np.dot(summary_embedding, trait_emb) / (
                    np.linalg.norm(summary_embedding) * np.linalg.norm(trait_emb)
                )
                similarities.append(float(similarity))
            
            # 유사도를 0~1 범위로 정규화
            similarities = np.array(similarities)
            similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
            
            category_embeddings[category] = similarities.tolist()

        return category_embeddings

    except Exception as e:
        print(f"벡터 생성 중 오류 발생: {str(e)}")
        return {}

@app.post("/user/preference/from-chat")
def generate_preference_from_chat(req: UserChatRequest, db: Session = Depends(get_db)):
    try:
        # GPT로 요약 생성
        summary = generate_summary_with_gpt(req.history, "고객")
        if not summary:
            raise HTTPException(status_code=500, detail="요약 생성 실패")

        # SBERT로 벡터 생성
        vectors = generate_vectors_from_summary(summary)
        if not vectors:
            raise HTTPException(status_code=500, detail="벡터 생성 실패")

        # 사용자 찾기
        user = db.query(User).filter(User.email == req.email).first()
        if not user:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

        # 벡터만 반환 (DB 저장 없이)
        return JSONResponse(content={
            "message": "성향 벡터가 성공적으로 생성되었습니다.",
            "vectors": vectors
        })

    except Exception as e:
        print("❌ 오류 발생:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/caregiver/personality/from-chat")
def analyze_personality_from_chat(req: CaregiverChatRequest, db: Session = Depends(get_db)):
    try:
        # GPT로 요약 생성
        summary = generate_summary_with_gpt(req.history, "돌보미")
        if not summary:
            raise HTTPException(status_code=500, detail="요약 생성 실패")

        # SBERT로 벡터 생성
        vectors = generate_vectors_from_summary(summary)
        if not vectors:
            raise HTTPException(status_code=500, detail="벡터 생성 실패")

        return JSONResponse(content={"vectors": vectors})

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"예외 발생: {e}")

def calculate_dynamic_weights(history: List[str]) -> Dict[str, float]:
    """대화 내용을 분석하여 카테고리별 가중치를 동적으로 계산"""
    try:
        # 기본 가중치
        default_weights = {
            "parenting_style_vector": 0.25,
            "personality_traits_vector": 0.20,
            "communication_style_vector": 0.15,
            "caregiving_attitude_vector": 0.15,
            "handling_situations_vector": 0.10,
            "empathy_traits_vector": 0.10,
            "trust_time_vector": 0.05
        }

        # 카테고리별 키워드 정의
        category_keywords = {
            "parenting_style_vector": ["교육", "학습", "놀이", "안전", "보호", "애착", "신체활동", "자율성", "훈육"],
            "personality_traits_vector": ["성격", "성향", "외향적", "내향적", "감성", "이성", "융통성", "원칙", "꼼꼼", "자유", "유머", "침착"],
            "communication_style_vector": ["대화", "설명", "직관", "비언어", "지시", "소통"],
            "caregiving_attitude_vector": ["인내심", "적극적", "신뢰", "개입", "관찰", "독립"],
            "handling_situations_vector": ["갈등", "돌발상황", "계획", "유연", "대처"],
            "empathy_traits_vector": ["감정", "공감", "무던", "표현"],
            "trust_time_vector": ["시간", "엄수", "융통성", "신뢰"]
        }

        # 대화 내용을 하나의 문자열로 결합
        combined_text = " ".join(history)
        
        # SBERT를 사용하여 대화 내용 임베딩
        text_embedding = sbert_model.encode(combined_text)
        
        # 각 카테고리별 키워드 임베딩 및 유사도 계산
        category_scores = {}
        for category, keywords in category_keywords.items():
            # 키워드들을 SBERT로 임베딩
            keyword_embeddings = sbert_model.encode(keywords)
            
            # 각 키워드와의 코사인 유사도 계산
            similarities = []
            for keyword_emb in keyword_embeddings:
                similarity = np.dot(text_embedding, keyword_emb) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(keyword_emb)
                )
                similarities.append(float(similarity))
            
            # 최대 유사도 점수를 해당 카테고리의 점수로 사용
            category_scores[category] = max(similarities)

        # 점수 정규화
        total_score = sum(category_scores.values())
        if total_score > 0:
            normalized_scores = {k: v/total_score for k, v in category_scores.items()}
        else:
            normalized_scores = default_weights

        # 기본 가중치와 정규화된 점수를 결합 (70:30 비율)
        final_weights = {
            category: 0.7 * default_weights[category] + 0.3 * normalized_scores[category]
            for category in default_weights.keys()
        }

        # 가중치 정규화 (합이 1이 되도록)
        total_weight = sum(final_weights.values())
        final_weights = {k: v/total_weight for k, v in final_weights.items()}

        return final_weights

    except Exception as e:
        print(f"가중치 계산 중 오류 발생: {str(e)}")
        return default_weights

def normalize_vectors(vectors: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """각 카테고리별 벡터를 정규화"""
    normalized = {}
    for category, vec in vectors.items():
        vec_array = np.array(vec)
        norm = np.linalg.norm(vec_array)
        if norm > 0:
            normalized[category] = (vec_array / norm).tolist()
        else:
            normalized[category] = vec
    return normalized

@app.post("/recommend/caregiver")
async def recommend_caregiver(req: RecommendationRequest, db: Session = Depends(get_db)):
    try:
        # 요청 데이터 로깅
        print("📨 받은 JSON 데이터:")
        print("vectors:", json.dumps(req.vectors, indent=2, ensure_ascii=False))
        print("history:", json.dumps(req.history, indent=2, ensure_ascii=False))
        
        # 1. 사용자 찾기
        #user = db.query(User).filter(User.email == req.email).first()
        #if not user:
        #    raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

        # 2. 벡터 데이터 확인
        vectors = req.vectors
        if not vectors:
            raise HTTPException(status_code=400, detail="벡터 데이터가 없습니다.")

        # 3. 가중치 계산
        weights = calculate_dynamic_weights(req.history)


        # 4. 모든 돌보미 가져오기
        caregivers = db.query(Caregiver).all()
        similarities = []

        # 5. 각 돌보미와의 유사도 계산
        for caregiver in caregivers:
            # 돌보미 벡터 생성 및 정규화
            caregiver_vectors = {
                "parenting_style_vector": json.loads(caregiver.parenting_style_vector),
                "personality_traits_vector": json.loads(caregiver.personality_traits_vector),
                "communication_style_vector": json.loads(caregiver.communication_style_vector),
                "caregiving_attitude_vector": json.loads(caregiver.caregiving_attitude_vector),
                "handling_situations_vector": json.loads(caregiver.handling_situations_vector),
                "empathy_traits_vector": json.loads(caregiver.empathy_traits_vector),
                "trust_time_vector": json.loads(caregiver.trust_time_vector)
            }
            caregiver_vectors = normalize_vectors(caregiver_vectors)

            # 각 카테고리별 코사인 유사도 계산
            category_similarities = {}
            for category in vectors.keys():
                user_vec = np.array(vectors[category])
                care_vec = np.array(caregiver_vectors[category])
                
                # 코사인 유사도 계산
                similarity = np.dot(user_vec, care_vec) / (
                    np.linalg.norm(user_vec) * np.linalg.norm(care_vec)
                )
                category_similarities[category] = float(similarity)

            # 전체 유사도 계산 (동적 가중치 적용)
            total_similarity = sum(
                category_similarities[cat] * weights[cat]
                for cat in weights.keys()
            )

            similarities.append((total_similarity, caregiver, category_similarities))

        # 6. 상위 3명 추출
        top3 = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

        # 7. 결과 포맷팅
        result = []
        for total_sim, caregiver, cat_sims in top3:
            # GPT로 추천 이유 생성
            explanation = generate_recommendation_explanation(caregiver, vectors, cat_sims)
            
            result.append({
                "name": caregiver.user.username,
                "age": caregiver.age,
                "total_similarity": round(total_sim, 4),
                "category_similarities": {
                    cat: round(sim, 4)
                    for cat, sim in cat_sims.items()
                },
                "explanation": explanation
            })

        return {"recommendations": result}

    except Exception as e:
        print("❌ 오류 발생:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ✅ 벡터 DB 경로 설정 (vectorDB 폴더에 chroma.sqlite3 포함)
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vectorDB")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ 벡터 DB 로드
vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)

@app.post("/ask/")
def ask_question(req: QueryRequest):
    try:
        # 🔍 관련 문서 검색
        docs = vector_db.similarity_search(req.prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 🧠 GPT 응답 생성 (RAG + 다음 질문 유도)
        prompt = f"""
            너는 사용자와 감정적으로 연결되며, 육아와 관련된 다양한 주제에서 성향을 파악하고 적절한 돌보미 매칭을 도와주는 상담사야.
            아래 참고 정보를 활용해서 사용자의 질문에 정확하고 따뜻하게 답변해줘. 아이의 성향을 얘기하면 그에 적합한 돌보미를 매칭해주는 거야. 
            그리고 성향을 잘 파악할 수 있도록 항상 자연스럽게 다음 질문을 이어가도록 해줘.
            문맥에 따라 사용자가 힘들어하거나 심리적으로 불안정하면 위로해줘. 하지만 제일 우선해야할 것은 돌보미 매칭을 위한 성향 파악이야.
            결과적으로 너는 돌보미 매칭을 위해 존재하는거야. 돌보미 매칭이 최우선이야. 돌보미=보모=보호자
            [참고 정보]
            {context}

            [질문]
            {req.prompt}

            [답변]
            """
        
        gpt_result = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "너는 섬세하고 따뜻한 육아 상담사야."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = gpt_result.choices[0].message.content.strip()
        return {"answer": answer}

    except Exception as e:
        logger.error(f"/ask/ 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {e}")


    
@app.get("/parent/info")
def get_parent_info(email: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.parent:
        raise HTTPException(status_code=404, detail="부모 정보가 없습니다.")
    
    parent = user.parent
    return {
        "children_count": parent.children_count,
        "is_multi_child": parent.is_multi_child,
        "income_type": parent.income_type,
        "preferred_service": parent.service_type_name,
        "last_calculated_fee": parent.last_calculated_fee,
        "hours": parent.hours,
        "hourly_fee": parent.hourly_fee,
        "total_fee": parent.total_fee,
        "gov_support_fee": parent.gov_support_fee
    }

# ✅ 돌보미 대화 → GPT 응답 생성 (RAG)
@app.post("/rag/")
def caregiver_rag_response(req: QueryRequest):
    try:
        docs = vector_db.similarity_search(req.prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
                당신은 아이돌보미 지원자의 성향을 파악하는 심리 상담사입니다.
                지원자가 자신의 경험이나 가치관, 아이 돌봄 방식에 대해 이야기하면,
                그에 어울리는 성격 특성을 유추하고 자연스럽게 이어질 수 있는 질문을 던져주세요.

                [지원자]
                {req.prompt}

                """

        gpt_result = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "너는 섬세하고 따뜻한 육아 상담사야."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = gpt_result.choices[0].message.content.strip()
        return {"answer": answer}

    except Exception as e:
        logger.error(f"/rag/ 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {e}")

# ────────────────────────────────────────────────
# 🚩 Pydantic Models
# ────────────────────────────────────────────────

class VectorResponse(BaseModel):
    vectors: Dict[str, List[float]]

class VectorUpdateRequest(BaseModel):
    email: str
    parenting_style_vector: Optional[List[float]] = None
    personality_traits_vector: Optional[List[float]] = None
    communication_style_vector: Optional[List[float]] = None
    caregiving_attitude_vector: Optional[List[float]] = None
    handling_situations_vector: Optional[List[float]] = None
    empathy_traits_vector: Optional[List[float]] = None
    trust_time_vector: Optional[List[float]] = None

# ────────────────────────────────────────────────
# 🚩 성향 추출 API
# ────────────────────────────────────────────────
 
import traceback

# ────────────────────────────────────────────────
# 🚩 DB 저장 API
# ────────────────────────────────────────────────

@app.post("/caregiver/update-vectors")
def update_caregiver_vectors(req: VectorUpdateRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="해당 이메일의 사용자가 존재하지 않습니다.")

    caregiver = db.query(Caregiver).filter(Caregiver.user_id == user.id).first()
    if not caregiver:
        raise HTTPException(status_code=404, detail="해당 사용자는 돌보미가 아닙니다.")

    vector_fields = [
        "parenting_style_vector",
        "personality_traits_vector",
        "communication_style_vector",
        "caregiving_attitude_vector",
        "handling_situations_vector",
        "empathy_traits_vector",
        "trust_time_vector"
    ]

    for field in vector_fields:
        value = getattr(req, field)
        if value is not None:
            setattr(caregiver, field, json.dumps(value))

    db.commit()
    return {"message": "돌보미 성향 벡터가 성공적으로 업데이트되었습니다."}

def generate_recommendation_explanation(caregiver: Caregiver, user_vectors: Dict, category_similarities: Dict) -> str:
    """GPT를 사용하여 추천 이유를 감성적으로 설명"""
    # 전체 유사도 점수 계산
    total_similarity = sum(category_similarities.values()) / len(category_similarities)
    
    prompt = f"""
당신은 따뜻한 공감 능력을 가진 육아 전문가입니다.
아래 돌보미 정보를 바탕으로, 사용자의 감정·성향 벡터와의 유사도를 반영해
이 돌보미가 왜 적합한지 공감 어린 말투로 설명해주세요.

⚠️ 아래 **세 가지 형식**을 반드시 모두 포함하고, **항상 같은 구조와 톤**으로 작성해 주세요.
(1순위, 2순위, 3순위 모두 **같은 양식**을 따릅니다.)

---

👤 돌보미 정보:
- 이름: {caregiver.user.username}
- 나이: {caregiver.age}세

📊 유사도 점수: {total_similarity:.1%}

📝 출력 형식 (반드시 아래 항목 모두 포함):

1. 🧩 **전체적인 매칭 포인트 요약**  
   - 사용자와의 감정·성향에서 어떤 점이 특히 잘 맞는지 2~3줄로 요약

2. 💡 **카테고리별 강점 Top 3**  
   - parenting_style / empathy_traits / caregiving_attitude 등 주요 성향 중 상위 3개
   - 각 성향과 그 이유를 따뜻하고 설득력 있게 서술

3. 🎬 **돌봄 상황 시나리오 예시**  
   - 실제 아이를 돌보는 상황에서 이 돌보미가 어떤 반응을 보일지
   - 사용자에게 위로와 신뢰를 줄 수 있는 **따뜻한 조언**을 덧붙여주세요

---

모든 설명은 공감적이고 부드러운 말투로 작성해 주세요.
"""

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "당신은 따뜻한 마음을 가진 육아 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"추천 설명 생성 중 오류 발생: {str(e)}")
        return "추천 이유를 생성하는 중 오류가 발생했습니다."