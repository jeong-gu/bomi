from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite DB 경로 (users.db)
DATABASE_URL = "sqlite:///./users.db"

# SQLAlchemy Engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Session 클래스 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스 (모든 모델의 상속 대상)
Base = declarative_base()

# ✅ 이 함수가 꼭 필요합니다!
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
