import sqlite3
import hashlib
import random
import json
from datetime import datetime

# DB 연결
conn = sqlite3.connect("users.db")
cur = conn.cursor()

# 비밀번호 해시 함수
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 벡터 생성
def random_vector(n, base=0.1):
    vec = [base] * n
    chosen = random.sample(range(n), random.randint(1, 3))
    for i in chosen:
        vec[i] = round(random.uniform(0.5, 0.9), 2)
    return vec
caregiver_names = [
    "김은지", "박지민", "이서연", "최민수", "정하늘",
    "홍지훈", "윤수빈", "장예린", "강민재", "조유진",
    "서지호", "한도윤", "임다연", "노지후", "배소윤",
    "오하늘", "신유진", "양지우", "권민재", "백서윤",
    "문현우", "안지민", "유하늘", "남윤호", "조예린",
    "정예진", "하도현", "김지후", "송유리", "강서준"
]

# 돌보미 30명 생성
for i in range(30):
    username = caregiver_names[i]
    email = f"caregiver{i+1}@test.com"
    phone = f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    age = random.randint(25, 55)
    hashed_pw = hash_password("123")
    created_at = datetime.now().isoformat()

    # 요일: 무작위로 2~4개 선택
    available_days = random.sample(["월", "화", "수", "목", "금", "토", "일"], k=random.randint(2, 4))

    # 시간대: 무작위로 1~3개 슬롯 생성
    available_times = []
    for _ in range(random.randint(1, 3)):
        start = random.randint(1, 22)
        end = random.randint(start + 1, min(start + 4, 24))  # 최대 3시간 블럭
        available_times.append({"start": start, "end": end})

    # 나이 범위: 0.25 단위로 설정
    age_min = round(random.uniform(0.25, 6.0), 2)
    age_max = round(random.uniform(age_min, 12.0), 2)  # 최소값보다 크거나 같게

    # 1. users 테이블 삽입
    cur.execute("""
        INSERT INTO users (username, email, hashed_password, role, phone, age, created_at)
        VALUES (?, ?, ?, 'care', ?, ?, ?)
    """, (username, email, hashed_pw, phone, age, created_at))
    user_id = cur.lastrowid

    # 2. caregivers 테이블 삽입
    caregiver_data = {
        "user_id": user_id,
        "age": age,
        "embedding": None,
        "available_days": json.dumps(available_days, ensure_ascii=False),
        "available_times": json.dumps(available_times),
        "special_child": random.choice([0, 1]),
        "age_min": age_min,
        "age_max": age_max,
        "parenting_style_vector": json.dumps(random_vector(8)),
        "personality_traits_vector": json.dumps(random_vector(10)),
        "communication_style_vector": json.dumps(random_vector(5)),
        "caregiving_attitude_vector": json.dumps(random_vector(6)),
        "handling_situations_vector": json.dumps(random_vector(4)),
        "empathy_traits_vector": json.dumps(random_vector(4)),
        "trust_time_vector": json.dumps(random_vector(3)),
    }

    cur.execute("""
        INSERT INTO caregivers (
            user_id, age, embedding, available_days, available_times, special_child,
            age_min, age_max,
            parenting_style_vector, personality_traits_vector, communication_style_vector,
            caregiving_attitude_vector, handling_situations_vector,
            empathy_traits_vector, trust_time_vector
        ) VALUES (
            :user_id, :age, :embedding, :available_days, :available_times, :special_child,
            :age_min, :age_max,
            :parenting_style_vector, :personality_traits_vector, :communication_style_vector,
            :caregiving_attitude_vector, :handling_situations_vector,
            :empathy_traits_vector, :trust_time_vector
        )
    """, caregiver_data)

# 저장 후 종료
conn.commit()
conn.close()