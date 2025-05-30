# models.py

import enum
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Enum, ForeignKey,
    Text, LargeBinary, Float, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# ──────────────────────────────────────────────────────────────
# 1) 사용자 역할 Enum
# ──────────────────────────────────────────────────────────────
class UserRole(enum.Enum):
    고객 = "고객"
    돌보미 = "돌보미"

# ──────────────────────────────────────────────────────────────
# 2) User 모델
# ──────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, nullable=False)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role            = Column(Enum(UserRole), nullable=False)
    phone           = Column(String)
    age             = Column(Integer)
    created_at      = Column(DateTime, default=datetime.utcnow)

    # 관계 설정
    caregiver   = relationship("Caregiver",    back_populates="user",       uselist=False)
    parent      = relationship("Parent",       back_populates="user",       uselist=False)
    preferences = relationship("UserPreference", back_populates="user",     uselist=False)

# ──────────────────────────────────────────────────────────────
# 3) Caregiver 모델
# ──────────────────────────────────────────────────────────────
class Caregiver(Base):
    __tablename__ = "caregivers"

    id                     = Column(Integer,    primary_key=True)
    user_id                = Column(Integer,    ForeignKey("users.id"), unique=True)
    age                    = Column(Integer)

    # 성향 점수
    diligent               = Column(Float, default=0.0)
    sociable               = Column(Float, default=0.0)
    cheerful               = Column(Float, default=0.0)
    warm                   = Column(Float, default=0.0)
    positive               = Column(Float, default=0.0)
    observant              = Column(Float, default=0.0)

    # 임베딩
    embedding              = Column(LargeBinary, nullable=True)

    # 돌봄 조건
    available_days         = Column(String, nullable=True)  # e.g. "월,화,수"
    available_times        = Column(String, nullable=True)  # JSON 문자열
    special_child          = Column(Boolean, default=False)
    age_min                = Column(Float, nullable=True)
    age_max                = Column(Float, nullable=True)

    # 벡터 기반 추가 정보 (JSON 문자열)
    parenting_style_vector     = Column(Text, nullable=True)
    personality_traits_vector  = Column(Text, nullable=True)
    communication_style_vector = Column(Text, nullable=True)
    caregiving_attitude_vector = Column(Text, nullable=True)
    handling_situations_vector = Column(Text, nullable=True)
    empathy_traits_vector      = Column(Text, nullable=True)
    trust_time_vector          = Column(Text, nullable=True)

    # 관계 설정
    user    = relationship("User",    back_populates="caregiver")
    reviews = relationship(
        "Review",
        back_populates="caregiver",
        cascade="all, delete-orphan"
    )

# ──────────────────────────────────────────────────────────────
# 4) Parent 모델
# ──────────────────────────────────────────────────────────────
class Parent(Base):
    __tablename__ = "parents"

    id                  = Column(Integer, primary_key=True)
    user_id             = Column(Integer, ForeignKey("users.id"), unique=True)
    children_count      = Column(Integer, nullable=True)
    is_multi_child      = Column(Boolean, nullable=True)
    income_type         = Column(Integer, nullable=True)
    preferred_service   = Column(String, nullable=True)
    last_calculated_fee = Column(Integer, nullable=True)
    hours               = Column(Float,   nullable=True)
    hourly_fee          = Column(Integer, nullable=True)
    total_fee           = Column(Integer, nullable=True)
    gov_support_fee     = Column(Integer, nullable=True)
    service_type_name   = Column(String,  nullable=True)

    user = relationship("User", back_populates="parent")

# ──────────────────────────────────────────────────────────────
# 5) UserPreference 모델
# ──────────────────────────────────────────────────────────────
class UserPreference(Base):
    __tablename__ = "user_preferences"

    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, ForeignKey("users.id"), unique=True)
    preferred_style = Column(Text, nullable=True)
    embedding     = Column(LargeBinary, nullable=True)

    user = relationship("User", back_populates="preferences")

# ──────────────────────────────────────────────────────────────
# 6) Review 모델
# ──────────────────────────────────────────────────────────────
class Review(Base):
    __tablename__ = "reviews"

    id           = Column(Integer, primary_key=True, index=True)
    caregiver_id = Column(Integer, ForeignKey("caregivers.id"), index=True, nullable=False)
    parent_name  = Column(String, index=True, nullable=False)
    content      = Column(Text, nullable=False)
    timestamp    = Column(DateTime, default=datetime.utcnow, nullable=False)

    caregiver    = relationship("Caregiver", back_populates="reviews")
