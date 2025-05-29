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

    # GPT 임베딩 등 추가 정보
    embedding = Column(LargeBinary, nullable=True)

    # 돌봄 조건 정보
    available_days = Column(String, nullable=True)      # 예: "월,화,수"
    available_times = Column(String, nullable=True)     # 예: '[{"start":7,"end":17}]'
    special_child = Column(Boolean, default=False)
    age_min = Column(Float, nullable=True)
    age_max = Column(Float, nullable=True)

    # 벡터 기반 성향 정보 (JSON 문자열 형태)
    parenting_style_vector      = Column(Text, nullable=True)  # JSON: [0.1, 0.2, ...]
    personality_traits_vector   = Column(Text, nullable=True)
    communication_style_vector  = Column(Text, nullable=True)
    caregiving_attitude_vector  = Column(Text, nullable=True)
    handling_situations_vector  = Column(Text, nullable=True)
    empathy_traits_vector       = Column(Text, nullable=True)
    trust_time_vector           = Column(Text, nullable=True)

    # 관계
    user = relationship("User", back_populates="caregiver")
    reviews = relationship("Review", back_populates="caregiver", cascade="all, delete")


    
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

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True)
    caregiver_id = Column(Integer, ForeignKey("caregivers.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # 작성자 (부모)
    rating = Column(Float, nullable=True)  # 별점 (선택적)
    text = Column(Text, nullable=False)    # 후기 내용
    created_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    caregiver = relationship("Caregiver", back_populates="reviews")
    user = relationship("User")  # 역관계는 선택사항