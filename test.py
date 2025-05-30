import sqlite3
import hashlib
import random
import json
from datetime import datetime

# ▒▒ 0. 설정 ▒▒
DB_PATH   = "users.db"
TOTAL     = 100                                 # 생성할 돌보미 수
IMG_URL   = "https://randomuser.me/api/portraits/women/{}.jpg"

# ▒▒ 1. 유틸 함수 ▒▒
def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def varied_vector(n: int) -> list[float]:
    vec = [round(random.uniform(0.0, 0.2), 2) for _ in range(n)]
    for _ in range(random.randint(2, 4)):
        vec[random.randrange(n)] = round(random.uniform(0.5, 0.9), 2)
    return vec

def random_available_times() -> list[dict]:
    blocks = []
    for _ in range(random.randint(1, 3)):
        start = random.randint(1, 22)
        end   = min(start + random.randint(1, 3), 24)
        blocks.append({"start": start, "end": end})
    return blocks

# ▒▒ 2. 요일 분배 준비 ▒▒
DAY_ORDER = ["월","화","수","목","금","토","일"]
# 3~7일씩 균등 분배: 5가지 × 20명 = 100명
sizes_cycle = [3]*20 + [4]*20 + [5]*20 + [6]*20 + [7]*20
random.shuffle(sizes_cycle)

def make_days(k: int) -> list[str]:
    pick = random.sample(DAY_ORDER, k)
    return sorted(pick, key=lambda d: DAY_ORDER.index(d))

# ▒▒ 3. 이름 리스트 (100개) ▒▒
caregiver_names = [
    "김하은","박서연","이유진","최지우","정다은","한소연","장지윤","윤하린","조예은","홍지아",
    "서윤서","문하진","오유빈","신서윤","배지우","임하린","강지안","노수빈","백예린","양지민",
    "권서하","송다윤","하예진","남하윤","안지율","유서아","임다희","정윤하","김지안","박나윤",
    "이서영","최하린","정가은","한지우","조하윤","장서아","윤다인","홍세린","서지유","문수아",
    "배서윤","신유나","오하린","강윤서","노지안","백하은","양나연","권민서","송지우","하지윤",
    "김미소","박다연","이세린","최지현","정소미","한예린","장지은","윤서진","조하경","홍나현",
    "서민지","문연우","오지은","신라희","배소연","임민서","강유진","노지연","백서진","양가희",
    "권보미","송현아","하지은","남세연","안가은","유윤아","임선우","정수연","김도연","박시아",
    "이보민","최다희","정가윤","한라은","조서윤","장민주","윤지수","홍새봄","서다연","문지아",
    "배하윤","신다영","오지현","강유림","노예원","백시윤","양다민","권혜린","송시연","하지수"
]

# ▒▒ 4. DB 초기화 ▒▒
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

# image_url 컬럼이 없으면 추가
try:
    cur.execute("SELECT image_url FROM users LIMIT 1")
except sqlite3.OperationalError:
    cur.execute("ALTER TABLE users ADD COLUMN image_url TEXT")

# 기존 돌보미 데이터 삭제
cur.executescript("""
    DELETE FROM caregivers;
    DELETE FROM users;
""")
conn.commit()

# ▒▒ 5. 데이터 삽입 ▒▒
for i in range(TOTAL):
    name    = caregiver_names[i]
    email   = f"caregiver{i+1}@test.com"
    phone   = f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    age     = random.randint(25, 55)
    created = datetime.now().isoformat()
    img_url = IMG_URL.format(i % 100)  # 0~99 중 하나

    # 연령대 균등 분배: 0~2세, 3~5세, 6세 이상
    if i < TOTAL//3:
        age_min, age_max = round(random.uniform(0.25, 1.5), 2), round(random.uniform(1.6, 2.0), 2)
    elif i < 2*TOTAL//3:
        age_min, age_max = round(random.uniform(3.0, 4.0), 2), round(random.uniform(4.1, 5.0), 2)
    else:
        age_min, age_max = round(random.uniform(6.0, 7.0), 2), round(random.uniform(8.0, 12.0), 2)

    # --- users 테이블 삽입 ---
    cur.execute("""
        INSERT INTO users
          (username, email, hashed_password, role, phone, age, created_at, image_url)
        VALUES (?, ?, ?, '돌보미', ?, ?, ?, ?)
    """, (
        name, email, hash_password("123"),
        phone, age, created, img_url
    ))
    user_id = cur.lastrowid

    # --- caregivers 테이블 삽입 ---
    days_k = sizes_cycle[i]
    caregiver_data = {
        "user_id": user_id,
        "age": age,
        "embedding": None,
        "available_days": json.dumps(make_days(days_k), ensure_ascii=False),
        "available_times": json.dumps(random_available_times()),
        "special_child": random.choice([0, 1]),
        "age_min": age_min,
        "age_max": age_max,
        "parenting_style_vector": json.dumps(varied_vector(8)),
        "personality_traits_vector": json.dumps(varied_vector(10)),
        "communication_style_vector": json.dumps(varied_vector(5)),
        "caregiving_attitude_vector": json.dumps(varied_vector(6)),
        "handling_situations_vector": json.dumps(varied_vector(4)),
        "empathy_traits_vector": json.dumps(varied_vector(4)),
        "trust_time_vector": json.dumps(varied_vector(3)),
    }
    cur.execute("""
        INSERT INTO caregivers (
            user_id, age, embedding,
            available_days, available_times, special_child,
            age_min, age_max,
            parenting_style_vector, personality_traits_vector,
            communication_style_vector, caregiving_attitude_vector,
            handling_situations_vector, empathy_traits_vector,
            trust_time_vector
        ) VALUES (
            :user_id, :age, :embedding,
            :available_days, :available_times, :special_child,
            :age_min, :age_max,
            :parenting_style_vector, :personality_traits_vector,
            :communication_style_vector, :caregiving_attitude_vector,
            :handling_situations_vector, :empathy_traits_vector,
            :trust_time_vector
        )
    """, caregiver_data)

# ▒▒ 6. 저장 및 종료 ▒▒
conn.commit()
conn.close()
print(f"✅ {TOTAL}명의 돌보미가 균등 분포로 저장되었습니다.")
