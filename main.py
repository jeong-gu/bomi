import pickle
from typing import Optional
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
from models import Base, User, UserRole, Caregiver, UserPreference,Parent
from passlib.context import CryptContext
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List

# 환경 변수 로드
load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API 설정
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chat_model = "gpt-4o"

DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL)
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
    email: str


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
    "아이를 돌봐주시는 분에게 가장 바라는 ‘마음가짐’은 어떤 모습일까요?",
    "아이와 관계를 맺는 데 있어 중요하다고 느끼는 태도나 자세가 있다면요?",
    "돌봄은 한두 번보다 꾸준함이 중요하다고 하잖아요, 소중하게 생각하는 ‘꾸준함’이 있다면 어떤 걸까요?",
    "아이를 돌보는 과정에서 가족끼리 나누는 역할이나 분위기는 어떤 편인가요?",
    "누군가에게 중요한 일을 맡기게 될 때, 어떤 부분에서 신뢰를 느끼시나요?",
    "새로운 사람과 처음 만날 때, 어떤 인상을 받을 때 안심이 되시나요?",
    "아이에게 말을 걸거나 대화할 때, 어떤 말투나 표현을 좋아하세요?",
    "부탁한 일이 정확하게 이뤄졌을 때, 어떤 점에서 ‘잘 해주셨다’고 느끼시나요?",
    "특별히 바쁘거나 도움이 급히 필요할 때, 돌보미가 어떤 태도로 응대해주면 좋을까요?",
]

# @app.get("/questions/caregiver")
# def get_caregiver_questions():
#     return JSONResponse(content={"questions": caregiver_questions})

# def format_caregiver_prompt(username: str, answers: list[str]) -> str:
#     formatted_qa = "\n".join([
#         f"{i+1}. {q}\n답변: {answers[i]}"
#         for i, q in enumerate(caregiver_questions)
#     ])
#     return f"""
#         다음은 '{username}'라는 아이돌보미 지원자의 자기소개 및 질문 응답 내용입니다.
#         아래 내용을 바탕으로 지원자의 성격을 따뜻하고 신뢰감 있게 1문단으로 요약해 주세요:

#         {formatted_qa}

#         성격 요약:
#         """

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

@app.post("/user/preference/from-chat")
def generate_preference_from_chat(req: ChatPreferenceRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    try:
        dialogue = "\n".join(req.history[-5:])
        prompt = f"""
                    당신은 부모님의 성향을 분석해주는 육아 상담사입니다.
                    아래는 부모님과의 상담 대화 내용입니다. 해당 부모님의 육아 성향을 간결하고 따뜻하게 한 문단으로 요약해 주세요:

                    {dialogue}

                    육아 성향 요약:
                    """
        gpt_result = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "너는 부모의 육아 성향을 분석하는 따뜻한 상담사야."},
                {"role": "user", "content": prompt}
            ]
        )
        summary = gpt_result.choices[0].message.content.strip()
        embedding = embedding_model.embed_documents([summary])[0]
        pickled_embedding = pickle.dumps(embedding)

        # ✅ 기존 preference 존재 시 update
        existing_pref = db.query(UserPreference).filter_by(user_id=user.id).first()

        if existing_pref:
            existing_pref.preferred_style = summary
            existing_pref.embedding = pickled_embedding
            logger.info(f"[업데이트] user_id={user.id}")
        else:
            new_pref = UserPreference(
                user_id=user.id,
                preferred_style=summary,
                embedding=pickled_embedding
            )
            db.add(new_pref)
            logger.info(f"[삽입] user_id={user.id}")

        db.commit()
        return {"summary": summary}

    except Exception as e:
        db.rollback()
        logger.error(f"성향 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="성향 분석 중 오류 발생")

@app.post("/recommend/caregiver")
def recommend_caregiver(req: RecommendationRequest, db: Session = Depends(get_db)):
    import numpy as np

    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.preferences:
        raise HTTPException(status_code=404, detail="고객 성향 정보가 없습니다.")

    user_vec = pickle.loads(user.preferences.embedding)

    caregivers = db.query(Caregiver).all()
    similarities = []
    for caregiver in caregivers:
        care_vec = pickle.loads(caregiver.embedding)
        sim = np.dot(user_vec, care_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(care_vec))
        similarities.append((sim, caregiver))

    top3 = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

    result = [
        {
            "name": c.user.username,
            "age": c.age,
            "personality": c.personality,
            "similarity": round(sim, 4)
        }
        for sim, c in top3
    ]
    return {"recommendations": result}

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


from pydantic import BaseModel
from typing import List, Dict, Optional
import json, re
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session

# ────────────────────────────────────────────────
# 🚩 Pydantic Models
# ────────────────────────────────────────────────

class ChatHistoryRequest(BaseModel):
    email: str
    history: List[str]

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
 

@app.post("/caregiver/personality/from-chat", response_model=VectorResponse)
def analyze_personality_from_chat(data: ChatHistoryRequest, db: Session = Depends(get_db)):
    try:
        # 성향 카테고리 정의
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

        # GPT 입력 프롬프트 구성
        prompt = (
            "당신은 '돌보미 성향 자가진단 챗봇'입니다.\n"
            "사용자는 돌보미로서 본인의 돌봄 성향과 가치관을 이해하고자 자가진단을 수행하고 있습니다.\n\n"
            "이 대화는 실제 돌봄 현장에서 발생할 수 있는 상황을 가정한 15개의 역할극 질문에 대한 응답이며,\n"
            "지원자의 말투, 행동, 사고방식, 감정 표현 등을 기반으로 아래 7개 항목에 대해 0~1 사이의 수치로 성향을 정량적으로 분석해주세요:\n"
            "1) parenting_style_vector\n"
            "2) personality_traits_vector\n"
            "3) communication_style_vector\n"
            "4) caregiving_attitude_vector\n"
            "5) handling_situations_vector\n"
            "6) empathy_traits_vector\n"
            "7) trust_time_vector\n\n"
            "❗️이 분석은 어디까지나 자가진단을 위한 도구입니다.\n"
            "❗️'추천', '매칭', '평가', '권장', '연결', '도움이 된다' 등의 문맥은 절대 사용하지 마세요.\n"
            "❗️분석의 목적은 오직 지원자가 스스로를 더 잘 이해하도록 돕는 데 있습니다.\n"
            "⚠️ 반드시 아래 형식을 정확히 지켜 JSON으로 출력해주세요.\n"
            "⚠️ 각 항목의 길이는 고정이며, 수치는 0.0 ~ 1.0 사이여야 합니다.\n"
            "⚠️ 각 항목에 대해 판단 가능한 경우에만 'judged'를 true로 표시하고, 부족한 경우 false로 표시하세요.\n\n"
            "예시:\n"
            "{\n"
            "  \"vectors\": {\n"
            "    \"parenting_style_vector\": [0.1, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],\n"
            "    ...\n"
            "  },\n"
            "  \"judged\": {\n"
            "    \"parenting_style_vector\": true,\n"
            "    ...\n"
            "  }\n"
            "}\n\n"
            "🧠 아래는 돌보미 지원자의 자가진단 대화입니다:\n"
            + "\n".join(data.history)
        )

        # GPT 호출
        gpt_response = client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 돌보미 성향을 정량적으로 분석하는 자가진단 시스템입니다.\n"
                        "지원자는 돌보미로서 본인의 성향을 파악하고 이해하기 위해 역할극 기반 대화에 참여했습니다.\n\n"
                        "당신의 유일한 목적은 이 대화의 내용을 기반으로 7가지 항목에 대한 수치를 0~1 범위로 분석하고,\n"
                        "판단 가능한 항목만 'judged': true로 명시하는 것입니다.\n\n"
                        "절대 다음 표현을 사용하지 마세요: '추천', '매칭', '연결', '도움', '적절한 돌보미', '좋은 성향', '이런 유형에 맞는 아이' 등.\n"
                        "이 분석은 외부 목적이 아닌 오직 사용자의 자기 이해를 위한 자가진단입니다.\n"
                        "감정적인 반응 없이 분석가로서 일관되게, 정확하게 판단하세요."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        # JSON 응답 추출 및 정제
        raw = gpt_response.choices[0].message.content.strip()
        match = re.search(r"\{[\s\S]*?\}", raw)
        if not match:
            raise HTTPException(status_code=500, detail="GPT 응답에서 JSON을 찾을 수 없습니다.")

        parsed = json.loads(match.group())
        vectors = parsed.get("vectors", {})
        judged = parsed.get("judged", {})

        # 길이 보정 로직 적용
        result = {}
        for key, items in categories.items():
            expected_len = len(items)
            vec = vectors.get(key, [])

            if not isinstance(vec, list):
                vec = [0.0] * expected_len
            elif len(vec) < expected_len:
                vec += [0.0] * (expected_len - len(vec))
            elif len(vec) > expected_len:
                vec = vec[:expected_len]

            result[key] = vec

        return {
            "vectors": result,
            "judged": judged
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 실패: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예외 발생: {e}")




# ────────────────────────────────────────────────
# 🚩 DB 저장 API
# ────────────────────────────────────────────────

@app.post("/caregiver/update-vectors")
def update_caregiver_vectors(data: VectorUpdateRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
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
        value = getattr(data, field)
        if value is not None:
            setattr(caregiver, field, json.dumps(value))

    db.commit()
    return {"message": "돌보미 성향 벡터가 성공적으로 업데이트되었습니다."}
