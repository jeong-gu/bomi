from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey, Text, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

class UserRole(enum.Enum):
    고객 = "고객"
    돌보미 = "돌보미"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    phone = Column(String)
    age = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    caregiver = relationship("Caregiver", back_populates="user", uselist=False)
    preferences = relationship("UserPreference", back_populates="user", uselist=False)
    parent = relationship("Parent", back_populates="user", uselist=False)

from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship

class Caregiver(Base):
    __tablename__ = "caregivers"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    age = Column(Integer)

    # 기본 성향 feature (기존)
    diligent = Column(Float, default=0.0)
    sociable = Column(Float, default=0.0)
    cheerful = Column(Float, default=0.0)
    warm = Column(Float, default=0.0)
    positive = Column(Float, default=0.0)
    observant = Column(Float, default=0.0)

    # GPT 임베딩 등 추가 정보
    embedding = Column(LargeBinary, nullable=True)

    # 돌봄 조건 정보
    available_days = Column(String, nullable=True)      # 예: "월,화,수"
    available_times = Column(String, nullable=True)     # 예: '[{"start":7,"end":17}]'
    special_child = Column(Boolean, default=False)
    age_min = Column(Float, nullable=True)
    age_max = Column(Float, nullable=True)

    # ✅ 추가된 벡터 기반 성향 정보 (JSON 문자열 형태)
    parenting_style_vector        = Column(Text, nullable=True)  # JSON: [0.1, 0.2, ...]
    personality_traits_vector     = Column(Text, nullable=True)
    communication_style_vector    = Column(Text, nullable=True)
    caregiving_attitude_vector   = Column(Text, nullable=True)
    handling_situations_vector    = Column(Text, nullable=True)
    empathy_traits_vector         = Column(Text, nullable=True)
    trust_time_vector             = Column(Text, nullable=True)
#     categories = {
#     "parenting_style": [
#         "교육 중심", "정서 케어 중심", "자율성 중심", "훈육 중심",
#         "놀이 중심", "안전/보호 중심", "애착 중심", "신체 활동 중심"
#     ],
#     "personality_traits": [
#         "외향적", "내향적", "감성형", "이성형", "융통형", "원칙형",
#         "꼼꼼형", "자유형", "유머형", "침착형"
#     ],
#     "communication_style": [
#         "설명 중심", "직관 중심", "대화형", "비언어형", "지시형"
#     ],
#     "caregiving_attitude": [
#         "인내심 있는", "적극적인", "신뢰 중심", "개입형", "관찰형", "독립 유도형"
#     ],
#     "handling_situations": [
#         "갈등 중재형", "돌발 상황 대응형", "계획형", "유연 대응형"
#     ],
#     "empathy_traits": [
#         "감정 민감형", "공감 우선형", "무던한 형", "감정 표현형"
#     ],
#     "trust_time": [
#         "시간 엄수형", "융통성 있는", "신뢰 우선형"
#     ]
# }



    # 관계
    user = relationship("User", back_populates="caregiver")


    
class Parent(Base):
    __tablename__ = "parents"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)

    # 필수 정보 (이건 남겨두고)
    children_count = Column(Integer, nullable=True)
    is_multi_child = Column(Integer, nullable=True)
    income_type = Column(Integer, nullable=True)
    preferred_service = Column(String, nullable=True)

    # 나중에 입력되는 요금 관련 정보
    last_calculated_fee = Column(Integer, nullable=True)
    hours = Column(Float, nullable=True)
    hourly_fee = Column(Integer, nullable=True)
    total_fee = Column(Integer, nullable=True)
    gov_support_fee = Column(Integer, nullable=True)
    service_type_name = Column(String, nullable=True)

    # 관계
    user = relationship("User", back_populates="parent")
    
class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    preferred_style = Column(Text)     # 고객이 원하는 돌보미 스타일 텍스트 (예: 활발하고 다정한 사람)
    embedding = Column(LargeBinary)    # 고객의 스타일에 대한 벡터 표현 (GPT 임베딩 기반)

    user = relationship("User", back_populates="preferences")
