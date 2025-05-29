import streamlit as st
import requests
from streamlit_lottie import st_lottie
import sqlite3
import math
import re
import time

import requests

# ➕ 감정 점수 & 기록용 세션 상태 초기화
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = {"joy":0,"positive":0,"surprise":0,"anger":0,"sadness":0,"fear":0}
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []  # 슬픔 추이만 담을 경우




# 1) 전역 설정 & 세션 상태 초기화
############################################
st.set_page_config(page_title="나만의 아이돌, 보미", page_icon="☁️", layout="centered")

RAG_API_URL = "http://localhost:8005/rag/"

# --------------------------- 세션 기본값 ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "start"

# 고민 상담용
if "selected_category" not in st.session_state:
    st.session_state.selected_category = None
if "counseling_messages" not in st.session_state:
    st.session_state.counseling_messages = []

# 수다 떨기용
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ✅ 여기서 trigger_rerun을 반드시 초기화
if "trigger_rerun" not in st.session_state:
    st.session_state.trigger_rerun = False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "trigger_rerun" not in st.session_state:
    st.session_state.trigger_rerun = False
    
# ────────────────────────────────────────────────────
# 돌보미 회원가입 2단계 분기용 세션 초기화 (여기에 추가)
if "page" not in st.session_state:
    st.session_state.page = "start"
if "caregiver_reg_step" not in st.session_state:
    st.session_state.caregiver_reg_step = 0   # 0=기본정보, 1=성향진단, 2=조건설정
if "caregiver_temp_data" not in st.session_state:
    st.session_state.caregiver_temp_data = {}
# ────────────────────────────────────────────────────




# 2) 전역 CSS (앱 최상단에 한 번만 선언)
# ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css?family=Poppins:300,400,600&display=swap');

* {
  font-family: 'Poppins', sans-serif;
  margin: 0;
  padding: 0;
}

body {
  background: linear-gradient(135deg, #dbeeff 25%, #ffffff 100%) no-repeat center center fixed;
  background-size: cover;
}

/* 페이지 전체 배경 컨테이너 */
.block-container {
  background-color: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(10px);
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* 채팅 말풍선 */
.user-bubble {
  background-color: #cceeff;
  padding: 10px;
  border-radius: 10px;
  margin: 5px 0;
  max-width: 70%;
  margin-left: auto;
  box-shadow: 0 2px 4px rgba(0,0,0,0.15);
}
.assistant-bubble {
  background-color: #ffffff;
  padding: 10px;
  border-radius: 10px;
  margin: 5px 0;
  max-width: 70%;
  margin-right: auto;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* 헤더용 로그아웃 버튼 래퍼 */
.logout-btn {
  position: absolute;
  top: 20px;
  right: 20px;
}
/* Streamlit 버튼 태그 선택자 보정 */
.logout-btn .stButton > button {
    background: transparent !important;
    border: none !important;
    font-size: 1.5rem !important;
    cursor: pointer !important;
}
.logout-btn .stButton > button:hover {
    color: #ff6961 !important;
}
</style>
""", unsafe_allow_html=True)

############################################


def page_caregiver_personality():
    import streamlit as st
    import requests
    import json

    if "caregiver_self_messages" not in st.session_state:
        st.session_state.caregiver_self_messages = [{
            "role": "assistant",
            "content": (
                "안녕하세요\n"
                "돌보미로서의 나를 소개해보는 시간이에요.\n"
                "내가 아이들과 어떻게 지내는 편인지, 어떤 성향인지 자유롭게 이야기해 주세요!\n\n"
                "예시:\n"
                "- '약속은 꼭 지키려고 해요'\n"
                "- '아이 눈높이에 맞춰서 대화하려고 노력해요'\n"
                "- '장난꾸러기 아이들도 귀엽게 봐주는 편이에요'\n\n"
                "자유롭게 말씀해 주시면, 당신만의 따뜻한 돌봄 스타일을 분석해드릴게요 "
            )
        }]
    if "last_caregiver_self_input" not in st.session_state:
        st.session_state.last_caregiver_self_input = None
    if "waiting_for_trait_response" not in st.session_state:
        st.session_state.waiting_for_trait_response = False

    st.markdown("<h3 style='text-align:center;'>📝 돌보미 성향 자가진단</h3>", unsafe_allow_html=True)
    col_text, col_save = st.columns([5, 1])
    with col_save:
        if st.button("저장"):
            history = [m["content"] for m in st.session_state.caregiver_self_messages if m["role"] == "user"]
            if len(history) < 2:
                st.warning("성향 분석을 위해 최소 2개의 대화가 필요해요!")
            else:
                with st.spinner("성향 분석 및 저장 중..."):
                    try:
                        res1 = requests.post(
                            "http://localhost:8005/caregiver/personality/from-chat",
                            json={"email": st.session_state.user_email, "history": history}
                        )
                        res1.raise_for_status()
                        # GPT 분석 결과 받기
                        vectors = res1.json().get("vectors", {})

                        categories = [
                            "parenting_style_vector",
                            "personality_traits_vector",
                            "communication_style_vector",
                            "caregiving_attitude_vector",
                            "handling_situations_vector",
                            "empathy_traits_vector",
                            "trust_time_vector"
                        ]

                        category_to_length = {
                            "parenting_style_vector": 8,
                            "personality_traits_vector": 10,
                            "communication_style_vector": 5,
                            "caregiving_attitude_vector": 6,
                            "handling_situations_vector": 4,
                            "empathy_traits_vector": 4,
                            "trust_time_vector": 3
                        }

                        # 누락된 항목은 0.0으로 채움
                        for cat in categories:
                            if cat not in vectors:
                                vectors[cat] = [0.0] * category_to_length[cat]

                        # 서버에 업데이트 요청
                        res2 = requests.post(
                            "http://localhost:8005/caregiver/update-vectors",
                            json={"email": st.session_state.user_email, **vectors}
                        )
                        res2.raise_for_status()

                        # 성공 메시지 출력
                        st.success("성향 벡터가 성공적으로 저장되었어요! \n홈 화면으로 이동합니다.")
                        st.session_state.page = "start"
                        st.rerun()


                    except requests.exceptions.RequestException as e:
                        st.error(f"서버 요청 중 오류 발생: {e}")
                    except Exception as e:
                        st.error(f"예기치 않은 오류: {e}")

    # 채팅 렌더링
    html = '<div class="chat-container">'
    if st.session_state.waiting_for_trait_response:
        html += '<div class="loading-bubble">답변 생성 중...</div>'
    for msg in st.session_state.caregiver_self_messages:
        cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        tag = "Q:" if msg["role"] == "user" else "A:"
        html += f'<div class="{cls}"><strong>{tag}</strong> {msg["content"]}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # 입력창
    def _on_enter():
        ui = st.session_state.caregiver_self_input
        if not ui or ui == st.session_state.last_caregiver_self_input:
            return
        st.session_state.caregiver_self_messages.append({"role": "user", "content": ui})
        st.session_state.last_caregiver_self_input = ui
        st.session_state.waiting_for_trait_response = True
        st.session_state.caregiver_self_input = ""

    st.text_input("", key="caregiver_self_input",
                  placeholder="성향에 대해 말씀해주세요!",
                  on_change=_on_enter)

    # GPT 응답 받기
    if st.session_state.waiting_for_trait_response:
        with st.spinner("답변 생성 중..."):
            resp = requests.post(
                "http://localhost:8005/caregiver/ask", 
                json={"prompt": st.session_state.last_caregiver_self_input,
                      "category": "caregiver_personality"}
            )
        answer = resp.json().get("answer", "응답이 없습니다.")
        st.session_state.caregiver_self_messages.append(
            {"role": "assistant", "content": answer}
        )
        st.session_state.waiting_for_trait_response = False
        st.rerun()








############################################
# 4) 페이지별 함수
############################################
def page_start():
    import streamlit as st
    import requests

    st.markdown("<h2 style='text-align:center;'>👤 로그인 또는 회원가입</h2>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔐 로그인", "📝 회원가입"])

    # ✅ 로그인 탭
    with tab1:
        login_email = st.text_input("이메일", key="login_email")
        login_pw = st.text_input("비밀번호", type="password", key="login_pw")

        if st.button("로그인"):
            try:
                response = requests.post("http://localhost:8005/login", json={
                    "email": login_email,
                    "password": login_pw
                })
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.logged_in = True
                    st.session_state.user_email = login_email
                    st.session_state.user_name = data["username"]
                    st.session_state.user_role = data["role"]
                    st.session_state.phone = data["phone"]
                    st.success("로그인 성공!")
                    st.session_state.page = "home"
                    st.rerun()
                else:
                    st.error(response.json()["detail"])
            except Exception as e:
                st.error(f"서버 오류: {e}")

    # ✅ 회원가입 탭
    with tab2:
        reg_role = st.selectbox("역할 선택", options=["고객", "돌보미"], key="reg_role_select")

        # 선택 즉시 상태 초기화 및 rerun
        if "last_role" not in st.session_state or st.session_state.last_role != reg_role:
            st.session_state.last_role = reg_role

            # 돌보미 선택 시 필요한 초기화
            if reg_role == "돌보미":
                st.session_state.caregiver_reg_step = 1
                st.session_state.caregiver_temp_data = {}
            st.rerun()  # 반드시 rerun

        # ───────────── 고객 회원가입 ─────────────
        if reg_role == "고객":
            st.subheader("🧍 고객 회원가입")
            reg_username = st.text_input("이름", key="cust_username")
            reg_email = st.text_input("이메일", key="cust_email")
            reg_pw = st.text_input("비밀번호", type="password", key="cust_pw")
            reg_age = st.number_input("나이", min_value=10, max_value=100, key="cust_age")
            reg_phone = st.text_input("전화번호", placeholder="예: 010-1234-5678", key="cust_phone")

            if st.button("고객 회원가입"):
                payload = {
                    "username": reg_username,
                    "email": reg_email,
                    "password": reg_pw,
                    "age": int(reg_age),
                    "phone": reg_phone,
                    "role": "고객"
                }
                try:
                    response = requests.post("http://localhost:8005/register", json=payload)
                    if response.status_code == 200:
                        st.success("회원가입 성공! 로그인 해주세요.")
                    else:
                        st.error(response.json()["detail"])
                except Exception as e:
                    st.error(f"서버 오류: {e}")

        # ───────────── 돌보미 회원가입 ─────────────
        elif reg_role == "돌보미":
            # ✅ 상태 초기화
            if "caregiver_reg_step" not in st.session_state:
                st.session_state.caregiver_reg_step = 1
            if "caregiver_temp_data" not in st.session_state:
                st.session_state.caregiver_temp_data = {}

            # Step 1
            if st.session_state.caregiver_reg_step == 1:
                st.subheader("🍼 돌보미 회원가입 - 1단계")
                reg_username = st.text_input("이름", key="care_username")
                reg_email = st.text_input("이메일", key="care_email")
                reg_pw = st.text_input("비밀번호", type="password", key="care_pw")
                reg_age = st.number_input("나이", min_value=18, max_value=100, key="care_age")
                reg_phone = st.text_input("전화번호", placeholder="예: 010-1234-5678", key="care_phone")

                if st.button("다음"):
                    st.session_state.caregiver_temp_data.update({
                        "username": reg_username,
                        "email": reg_email,
                        "password": reg_pw,
                        "age": int(reg_age),
                        "phone": reg_phone,
                    })
                    st.session_state.caregiver_reg_step = 2
                    st.rerun()

            # Step 2
            elif st.session_state.caregiver_reg_step == 2:
                st.subheader(" 돌봄이 회원가입 - 2단계")
                st.markdown("돌불 조건을 설정해주세요.")

                days = ["월", "화", "수", "목", "금", "토", "일"]
                selected_days = []
                select_all = st.checkbox("모든 요일 선택", key="select_all_days")
                cols = st.columns(7)
                for i, day in enumerate(days):
                    is_checked = select_all or (day in st.session_state.caregiver_temp_data.get("available_days", []))
                    if cols[i].checkbox(day, value=is_checked, key=f"day_{day}"):
                        selected_days.append(day)

                # 시간아웃 (일반 버튼)
                st.markdown("<h4 style='color: #2c3e50;'>시간대 추가</h4>", unsafe_allow_html=True)
                if "time_slots" not in st.session_state:
                    st.session_state.time_slots = st.session_state.caregiver_temp_data.get("available_times", [])

                def add_time_slot():
                    st.session_state.time_slots.append({"start": 1, "end": 1})

                if st.button("⏰ 시간대 추가", key="add_time"):
                    add_time_slot()

                for i, slot in enumerate(st.session_state.time_slots):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        slot["start"] = col1.selectbox("시작 시간", range(1, 25), index=slot["start"]-1, key=f"start_{i}")
                    with col2:
                        slot["end"] = col2.selectbox("종료 시간", range(1, 25), index=slot["end"]-1, key=f"end_{i}")
                    with col3:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.time_slots.pop(i)
                            st.rerun()

                st.markdown("""
                <h4 style='color: #2c3e50;'>특수아동 가능여부</h4>
                """, unsafe_allow_html=True)
                special_child = st.radio("", ["O", "X"], horizontal=True, key="special_child")

                st.markdown("""
                <h4 style='color: #2c3e50;'>돌불 가능 연병대</h4>
                """, unsafe_allow_html=True)
                age_range = st.slider("", 0.25, 12.0, st.session_state.caregiver_temp_data.get("age_range", (0.25, 12.0)), step=0.25, format="%.2f")

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("\u25c0 \uc774전"):
                        st.session_state.caregiver_temp_data.update({
                            "available_days": selected_days,
                            "available_times": st.session_state.time_slots,
                            "special_child": special_child,
                            "age_range": list(age_range)
                        })
                        st.session_state.caregiver_reg_step = 1
                        st.rerun()

                # ✅ 돌보미 회원가입 - 2단계 처리
                with col3:
                    if st.button("회원가입"):
                        if not selected_days:
                            st.warning("가능한 요일을 하나 이상 선택해주세요.")
                            st.stop()
                        if not st.session_state.time_slots:
                            st.warning("가능한 시간대를 하나 이상 추가해주세요.")
                            st.stop()

                        # 조건 저장
                        st.session_state.caregiver_temp_data.update({
                            "available_days": selected_days,
                            "available_times": st.session_state.time_slots,
                            "special_child": special_child == "O",
                            "age_range": list(age_range),
                            "role": "돌보미"
                        })

                        # backend 형식에 맞게 conditions 필드 생성
                        caregiver_info = st.session_state.caregiver_temp_data
                        conditions = {
                            "days": caregiver_info["available_days"],
                            "times": caregiver_info["available_times"],
                            "special": caregiver_info["special_child"],
                            "age_min": caregiver_info["age_range"][0],
                            "age_max": caregiver_info["age_range"][1]
                        }

                        payload = {
                            "username": caregiver_info["username"],
                            "email": caregiver_info["email"],
                            "password": caregiver_info["password"],
                            "role": "돌보미",
                            "age": caregiver_info["age"],
                            "phone": caregiver_info["phone"],
                            "conditions": conditions
                        }

                        try:
                            res = requests.post("http://localhost:8005/register", json=payload)
                            res.raise_for_status()
                            st.session_state.user_email = payload["email"]

                            st.success("회원가입이 완료되었습니다! 이제 성향 자가진단을 시작해볼게요.")
                            st.session_state.page = "caregiver_personality"
                            st.rerun()

                        except requests.exceptions.RequestException as e:
                            st.error(f"회원가입 중 오류 발생: {e}")
                            st.stop()

 

def page_caregiver_home():
    # 1) 로그인 확인
    if not st.session_state.get("logged_in"):
        st.warning("로그인이 필요합니다.")
        st.session_state.page = "start"
        st.rerun()

    user_name = st.session_state.get("user_name", "사용자")

    # ────────────────────────────────────────────────
    # 이 함수 전용 CSS
    # ────────────────────────────────────────────────
    st.markdown("""
    <style>
      /* 페이지 전체를 꽉 채우는 컨테이너 */
      .block-container {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
      }
      /* 헤더 로그아웃 버튼 스타일 */
      .logout-btn > button {
        background: transparent !important;
        border: none !important;
        font-size: 1.5rem !important;
        cursor: pointer !important;
      }
      .logout-btn > button:hover {
        color: #ff6961 !important;
      }
      /* 메뉴 버튼 고정 크기 + 간격 */
      .menu-btns .stButton > button {
        width: 140px !important;
        margin: 0 10px !important;
        white-space: nowrap !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────
    # 2) 헤더: 로그아웃 버튼 + 환영 문구
    # ────────────────────────────────────────────────
    st.markdown(
        '<div class="logout-btn" style="width:100%; text-align:right; padding:10px 0;">',
        unsafe_allow_html=True
    )
    if st.button("⏻", key="logout_caregiver"):
        for k in ["logged_in","user_email","user_role","user_name"]:
            st.session_state.pop(k, None)
        st.session_state.page = "start"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f"<h3 style='width:100%; text-align:left; padding-bottom:20px;'>"
        f"환영해요, <strong>{user_name}</strong> 돌보미님.</h3>",
        unsafe_allow_html=True
    )

    # ────────────────────────────────────────────────
    # 3) 메뉴 버튼 2행 배치 (가운데 대칭)
    # ────────────────────────────────────────────────
    # 래퍼 div 에 클래스를 걸어 버튼 스타일 적용
    st.markdown('<div class="menu-btns" style="width:100%;">', unsafe_allow_html=True)

    # --- 1행: 3개 버튼 (정보용 / 성향분석 / 요금산정)
    row1 = st.columns(5)
    if row1[1].button("정보용"):
        st.session_state.page = "chat"; st.rerun()
    if row1[2].button("성향분석"):
        st.session_state.page = "caregiver_personality"; st.rerun()
    if row1[3].button("요금산정"):
        st.session_state.page = "pricing"; st.rerun()

    # --- 2행: 2개 버튼 (돌보미목록 / 조건설정)
    row2 = st.columns(5)
    if row2[1].button("돌보미목록"):
        st.session_state.page = "caregivers"; st.rerun()
    if row2[3].button("조건설정"):
        st.session_state.page = "caregiver_conditions"; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)






def page_parent_home():
    # 로그인 확인
    if not st.session_state.get("logged_in"):
        st.warning("로그인이 필요합니다.")
        st.session_state.page = "start"
        st.rerun()

    user_name = st.session_state.get("user_name", "사용자")

    # ── 헤더: 로그아웃 버튼 + 환영 메시지
    st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
    if st.button("⏻", key="logout_parent"):
        for k in ["logged_in","user_email","user_role","user_name"]:
            st.session_state.pop(k, None)
        st.session_state.page = "start"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f"<h3 style='text-align:left;'>환영해요, <strong>{user_name}</strong> 부모님.</h3>",
        unsafe_allow_html=True
    )

    # ── 메뉴 버튼 (4개)
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("정보용"):      st.session_state.page="chat";       st.rerun()
    if c2.button("추천용"):      st.session_state.page="recommend";  st.rerun()
    if c3.button("요금산정"):    st.session_state.page="pricing";    st.rerun()
    if c4.button("돌보미목록"):  st.session_state.page="caregivers"; st.rerun()





    
########################################
########################################
def page_pricing():
    
        # ✅ **상단 타이틀 바**
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        if st.button("◀", key="back_chat_btn"):
            st.session_state.page = "home"
            st.rerun()

    with col2:
        st.markdown(f"<h3 style='text-align: center;'>💵 요금 산정 설문</h3>", unsafe_allow_html=True)

    with col3:
        if st.button("🏠", key="home_chat_btn"):
            st.session_state.page = "home"
            st.rerun()
    
            
    # ✅ **CSS 스타일 수정 (입력창을 채팅창 내부에 완전히 포함)**
    st.markdown("""
    <style>
    .chat-container {
      width: 90%;
      max-width: 600px;
      height: 75vh;
      display: flex;
      flex-direction: column-reverse;
      overflow-y: auto;
      padding: 15px;
      background: white;
      margin: auto;
      border-radius: 15px;
      box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
      position: relative;
    }

    .chat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
      background: #ff9966;
      color: white;
      padding: 12px 16px;
      font-size: 18px;
      font-weight: bold;
      border-bottom: 2px solid #ff784f;
      border-radius: 8px 8px 0 0;
    }

    .chat-header h3 {
      flex-grow: 1;
      text-align: center;
      margin: 0;
    }

    .user-bubble {
      background: #d0f0ff;
      padding: 12px;
      border-radius: 20px;
      margin: 5px 0;
      max-width: 70%;
      margin-left: auto;
      text-align: left;
      box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);
    }

    .assistant-bubble {
      background: #ffcc99;
      padding: 12px;
      border-radius: 20px;
      margin: 5px 0;
      max-width: 70%;
      margin-right: auto;
      text-align: left;
      box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);
    }

    .loading-bubble {
      background: #fff2c7;
      padding: 12px;
      border-radius: 20px;
      margin: 5px 0;
      max-width: 70%;
      margin-right: auto;
      text-align: left;
      box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);
      font-weight: bold;
    }

    /* ✅ 입력창을 채팅창 내부 최하단에 고정 */
    .input-container {
      width: calc(100% - 30px);
      padding: 10px;
      background: white;
      border-top: 2px solid #ccc;
      display: flex;
      align-items: center;
      position: absolute;
      bottom: 0;
      left: 15px;
      border-radius: 0 0 15px 15px;
      box-shadow: 0px -2px 8px rgba(0, 0, 0, 0.1);
    }

    .input-container input {
      width: 100%;
      padding: 10px;
      border: none;
      outline: none;
      font-size: 16px;
      border-radius: 10px;
      background: #f1f3f4;
    }
    
    </style>
    """, unsafe_allow_html=True)

    # (1) 서비스 종류 옵션 정의
    SERVICE_OPTIONS = [
        (1, "시간제 기본형"),
        (2, "시간제 종합형"),
        (3, "영아종일제"),
        (4, "질병감염아동지원"),
        (5, "기관연계서비스"),
    ]

    with st.form("fee_form"):
        # 1) 서비스 종류
        sel = st.selectbox(
            "서비스 종류 선택",
            options=SERVICE_OPTIONS,
            format_func=lambda x: f"{x[0]}: {x[1]}",
            key="service_type"
        )
        service_type = sel[0]  # 선택된 튜플의 첫 번째 값

        # 2) 이용 시간
        hours = st.number_input(
            "이용 시간(시간 단위, 예: 3.5)", 
            min_value=0.5, max_value=24.0, value=1.0, step=0.5,
            key="hours"
        )

        # 3) 동시 돌봄 아동 수
        max_children = 5 if service_type == 5 else 3
        num_children = st.number_input(
            f"동시 돌봄 아동 수 (1-{max_children})", 
            min_value=1, max_value=max_children, value=1, step=1,
            key="children"
        )

        # 4) 소득 유형
        INCOME_OPTIONS = [
            (1, "가형 (중위소득 75% 이하)"),
            (2, "나형 (중위소득 120% 이하)"),
            (3, "다형 (중위소득 150% 이하)"),
            (4, "라형 (중위소득 200% 이하)"),
            (5, "마형 (중위소득 200% 초과)"),
        ]
        sel_inc = st.selectbox(
            "소득 유형 선택",
            options=INCOME_OPTIONS,
            format_func=lambda x: f"{x[0]}: {x[1]}",
            key="income_type"
        )
        income_type = sel_inc[0]

        # 5) 야간/휴일 여부
        is_night   = st.checkbox("야간 시간대 이용 (22시~06시)", key="is_night")
        is_holiday = st.checkbox("휴일(일요일/공휴일) 이용", key="is_holiday")

        # 6) 다자녀 여부 (소득 마형 제외)
        is_multi_child = False
        if income_type != 5:
            is_multi_child = st.checkbox("다자녀 가구 여부 (2명 이상)", key="is_multi")

        submitted = st.form_submit_button("다음 → 요금 계산")
        if submitted:
            st.session_state.fee_inputs = {
                "service_type": service_type,
                "hours":        hours,
                "children":     num_children,
                "income_type":  income_type,
                "is_night":     is_night,
                "is_holiday":   is_holiday,
                "is_multi":     is_multi_child,
            }
            st.session_state.page = "fee_result"
            st.rerun()
########################################
########################################


 


########################################
#############################################

def page_recommend_service():
    import requests

    # ─────────────────────────────────────────────────────────────────────
    # 1) 상태 변수 초기화
    # ─────────────────────────────────────────────────────────────────────
    if "recommend_messages" not in st.session_state:
        st.session_state.recommend_messages = [{
            "role": "assistant",
            "content": (
                "안녕하세요 😊 아이의 성향이나 돌봄에 대한 생각을 자유롭게 말씀해주세요!\n"
                "예: '우리 아이는 낯을 많이 가려요', '꼼꼼한 돌보미가 좋겠어요'"
            )
        }]
    if "last_recommend_input" not in st.session_state:
        st.session_state.last_recommend_input = None
    if "waiting_for_recommend_response" not in st.session_state:
        st.session_state.waiting_for_recommend_response = False
    if "recommend_done" not in st.session_state:
        st.session_state.recommend_done = False
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "current_emotion" not in st.session_state:
        st.session_state.current_emotion = {
            "joy": 0, "positive": 0, "surprise": 0,
            "anger": 0, "sadness": 0, "fear": 0
        }
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []

    # ─────────────────────────────────────────────────────────────────────
    # 2) CSS 스타일 (생략 없이 그대로)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .stButton > button {
        padding: 0.25rem 0.75rem !important;
        font-size: 0.9rem !important;
    }
    .chat-container {
        width: 90%;
        max-width: 600px;
        height: 70vh;
        display: flex;
        flex-direction: column;
        overflow-y: auto;
        padding: 15px;
        background: white;
        margin: auto;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        position: relative;
    }
    .user-bubble {
        background: #d0f0ff;
        padding: 12px;
        border-radius: 20px;
        margin: 5px 0;
        max-width: 70%;
        margin-left: auto;
        text-align: left;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    .assistant-bubble {
        background: #ffeb99;
        padding: 12px;
        border-radius: 20px;
        margin: 5px 0;
        max-width: 70%;
        margin-right: auto;
        text-align: left;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    .loading-bubble {
        background: #fff2c7;
        padding: 12px;
        border-radius: 20px;
        margin: 5px 0;
        max-width: 70%;
        margin-left: 0; /* 왼쪽 정렬 */
        text-align: left;
        font-weight: bold;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # 3) 네비게이션 바 (기존 그대로)
    # ─────────────────────────────────────────────────────────────────────
    col_back, col_title, col_btns = st.columns([1, 4, 2])
    with col_back:
        if st.button("◀", key="back_recommend"):
            st.session_state.page = "home"; st.rerun()
    with col_title:
        st.markdown("<h3 style='text-align:center;'>추천 서비스 챗봇</h3>", unsafe_allow_html=True)
    with col_btns:
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("추천", key="do_recommend"):
                # 🔧 문제 원인 해결: 문자열만 필터링
                history = [
                    str(m["content"]).strip()
                    for m in st.session_state.recommend_messages
                    if m["role"] == "user" and isinstance(m["content"], str) and m["content"].strip()
                ]
                if not history:
                    st.warning("최소 한 개 이상의 질문을 해주세요."); st.stop()

                try:
                    with st.spinner("👀 성향 분석 및 돌보미 추천 중..."):
                        # 성향 분석 요청
                        pref_resp = requests.post(
                            "http://localhost:8005/user/preference/from-chat",
                            json={"email": st.session_state.user_email, "history": history}
                        )
                        pref_resp.raise_for_status()

                        # 추천 요청
                        rec_resp = requests.post(
                            "http://localhost:8005/recommend/caregiver",
                            json={
                                "email": st.session_state.user_email,
                                "emotion": st.session_state.current_emotion
                            }
                        )
                        rec_resp.raise_for_status()

                        # 결과 저장 및 화면 전환
                        st.session_state.recommendations = rec_resp.json().get("recommendations", [])
                        st.session_state.page = "recommend_result"
                        st.rerun()

                except Exception as e:
                    st.error(f"🔴 추천 실패: {e}")
        with btn2:
            if st.button("🏠", key="home_recommend"):
                st.session_state.page = "home"; st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # 4) 채팅 메시지 영역 (로딩 배너 포함)
    # ─────────────────────────────────────────────────────────────────────
    messages_html = '<div class="chat-container" id="chat-messages">'
    for msg in st.session_state.recommend_messages:
        cls = "user-bubble" if msg["role"]=="user" else "assistant-bubble"
        tag = "Q:" if msg["role"]=="user" else "A:"
        messages_html += f'<div class="{cls}"><strong>{tag}</strong> {msg["content"]}</div>'
    # 여기에 바로 로딩 배너를 추가
    if st.session_state.waiting_for_recommend_response:
        messages_html += '<div class="loading-bubble">답변 생성 중...</div>'
    messages_html += '<div id="chat-end"></div></div>'
    messages_html += """
    <script>
      document.getElementById('chat-end')?.scrollIntoView({ behavior:'smooth' });
    </script>
    """
    st.markdown(messages_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # 5) 입력창 + 감정 콜 (Enter만으로 on_change 트리거)
    # ─────────────────────────────────────────────────────────────────────
    def _on_recommend_enter():
        ui = st.session_state.recommend_input
        if not ui or ui == st.session_state.last_recommend_input:
            return
        st.session_state.recommend_messages.append({"role":"user","content":ui})
        st.session_state.last_recommend_input = ui
        emo = requests.post(
            "http://localhost:8005/emotion/",
            json={"prompt": ui, "category":"general_chat"}
        ).json().get("scores",{})
        st.session_state.current_emotion = emo
        hist = st.session_state.emotion_history + [emo.get("sadness",0)]
        st.session_state.emotion_history = hist[-10:]
        st.session_state.waiting_for_recommend_response = True
        st.session_state.recommend_input = ""  # 입력창 초기화

    st.text_input("", key="recommend_input", placeholder="궁금한 점을 입력하세요!", on_change=_on_recommend_enter)

    # ─────────────────────────────────────────────────────────────────────
    # 6) AI 응답 호출
    # ─────────────────────────────────────────────────────────────────────
    if st.session_state.waiting_for_recommend_response:
        with st.spinner(""):
            lead = ""
            emo = st.session_state.current_emotion
            if emo.get("sadness",0)>0.5:    lead="[기분: 슬픔↑] "
            elif emo.get("anger",0)>0.5:    lead="[기분: 분노↑] "
            resp = requests.post(
                RAG_API_URL,
                json={"prompt": lead + st.session_state.last_recommend_input, "category":"general_chat"}
            )
        answer = resp.json().get("answer","🚨 응답 없음.")
        st.session_state.recommend_messages.append({"role":"assistant","content":answer})
        st.session_state.waiting_for_recommend_response = False
        st.rerun()


######################################################
#############################################

def page_recommend_result():
    st.markdown("<h3 style='text-align:center;'>🧡 돌보미 추천 결과</h3>", unsafe_allow_html=True)

    recommendations = st.session_state.get("recommendations", [])
    if not recommendations:
        st.warning("아직 추천된 돌보미가 없습니다. 먼저 상담 챗봇에서 추천을 받아보세요!")
        if st.button("◀ 추천 챗봇으로 돌아가기"):
            st.session_state.page = "recommend"
            st.rerun()
        return

    for r in recommendations:
        st.markdown(f"""
        ---
        **👩‍🍼 {r['name']}** (나이: {r['age']}세)  
        📝 {r['personality']}  
        💡 유사도: **{r['similarity'] * 100:.1f}%**
        """)

    if st.button("◀ 다시 상담 챗봇으로"):
        st.session_state.page = "recommend"
        st.rerun()




########################################
def page_chat_talk():
    # ✅ 상태 변수 설정
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": (
                "안녕하세요 😊\n 무엇이든 물어보세요!!"
            )
        }]  # 수다 대화 상태 변수 추가
        
    if "last_chat_input" not in st.session_state:
        st.session_state.last_chat_input = None  
    if "waiting_for_chat_response" not in st.session_state:
        st.session_state.waiting_for_chat_response = False

    # ✅ **CSS 스타일 수정 (입력창을 채팅창 내부에 완전히 포함)**
    st.markdown("""
    <style>
    .chat-container {
      width: 90%;
      max-width: 600px;
      height: 75vh;
      display: flex;
      flex-direction: column-reverse;
      overflow-y: auto;
      padding: 15px;
      background: white;
      margin: auto;
      border-radius: 15px;
      box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
      position: relative;
    }

    .chat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
      background: #ffcc66;
      color: white;
      padding: 12px 16px;
      font-size: 18px;
      font-weight: bold;
      border-bottom: 2px solid #ffb347;
      border-radius: 8px 8px 0 0;
    }

    .chat-header h3 {
      flex-grow: 1;
      text-align: center;
      margin: 0;
    }

    .user-bubble {
      background: #d0f0ff;
      padding: 12px;
      border-radius: 20px;
      margin: 5px 0;
      max-width: 70%;
      margin-left: auto;
      text-align: left;
      box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);
    }

    .assistant-bubble {
      background: #ffeb99;
      padding: 12px;
      border-radius: 20px;
      margin: 5px 0;
      max-width: 70%;
      margin-right: auto;
      text-align: left;
      box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);
    }

    .loading-bubble {
      background: #fff2c7;
      padding: 12px;
      border-radius: 20px;
      margin: 5px 0;
      max-width: 70%;
      margin-right: auto;
      text-align: left;
      box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);
      font-weight: bold;
    }

    /* ✅ 입력창을 채팅창 내부 최하단에 고정 */
    .input-container {
      width: calc(100% - 30px);
      padding: 10px;
      background: white;
      border-top: 2px solid #ccc;
      display: flex;
      align-items: center;
      position: absolute;
      bottom: 0;
      left: 15px;
      border-radius: 0 0 15px 15px;
      box-shadow: 0px -2px 8px rgba(0, 0, 0, 0.1);
    }

    .input-container input {
      width: 100%;
      padding: 10px;
      border: none;
      outline: none;
      font-size: 16px;
      border-radius: 10px;
      background: #f1f3f4;
    }
    
    </style>
    """, unsafe_allow_html=True)

    # ✅ **상단 타이틀 바**
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        if st.button("◀", key="back_chat_btn"):
            st.session_state.page = "home"
            st.rerun()

    with col2:
        st.markdown(f"<h3 style='text-align: center;'> 정보의 바다!</h3>", unsafe_allow_html=True)

    with col3:
        if st.button("🏠", key="home_chat_btn"):
            st.session_state.page = "home"
            st.rerun()

    # ✅ **채팅 메시지 컨테이너 (입력창 포함)**
    messages_html = '<div class="chat-container" id="chat-messages">'

    # ✅ "🐝 답변 생성 중..."을 조건부로 표시
    if st.session_state.waiting_for_chat_response:
        messages_html += '<div class="loading-bubble">🐝 답변 생성 중...</div>'

    # ✅ 기존 메시지 렌더링
    for msg in reversed(st.session_state.chat_messages):
        if msg["role"] == "user":
            messages_html += f'<div class="user-bubble"><strong>Q:</strong> {msg["content"]}</div>'
        else:
            messages_html += f'<div class="assistant-bubble"><strong>A:</strong> {msg["content"]}</div>'

    messages_html += '</div>'
    st.markdown(messages_html, unsafe_allow_html=True)

    # ✅ **입력창을 채팅창 내부 최하단에 고정 (단일 입력창 유지)**
    user_q = st.text_input(
        "자유롭게 수다를 떨어보세요!", 
        key="chat_input", 
        label_visibility="collapsed"
    )

    # ✅ **질문 입력 처리**
    if user_q and user_q != st.session_state.last_chat_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        st.session_state.waiting_for_chat_response = True
        st.session_state.last_chat_input = user_q
        st.rerun()

    # ✅ **AI 응답 생성 (자동 호출)**
    if st.session_state.waiting_for_chat_response:
        with st.spinner("🐝 답변 생성 중..."):
            try:
                resp = requests.post(
                    RAG_API_URL,
                    json={"prompt": st.session_state.chat_messages[-1]["content"], "category": "general_chat"}
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "🚨 응답 없음.")
            except requests.exceptions.RequestException as e:
                answer = f"오류 발생: {str(e)}"

        # ✅ "답변 생성 중..." 제거 후 실제 응답 추가
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.session_state.waiting_for_chat_response = False
        st.rerun()

########################################



########################################

import sqlite3
import math
import re

# 요일 구간을 실제 요일 리스트로 변환하는 함수
def expand_days(availability_text):
    day_order = ['월', '화', '수', '목', '금', '토', '일']
    days = set()

    # '주말' 포함 처리
    if '주말' in availability_text:
        days.update(['토', '일'])

    # '화~토' 같은 범위 처리
    matches = re.findall(r'([월화수목금토일])\~([월화수목금토일])', availability_text)
    for start, end in matches:
        si = day_order.index(start)
        ei = day_order.index(end)
        if si <= ei:
            days.update(day_order[si:ei+1])
        else:
            days.update(day_order[si:] + day_order[:ei+1])

    # 개별 요일 포함도 추가
    for d in day_order:
        if d in availability_text:
            days.add(d)

    return days
def page_fee_result():
    """💵 요금 계산 결과 (fee.py 알고리즘 반영)"""
    inputs = st.session_state.get("fee_inputs")
    if not inputs:
        st.error("요금 산정 정보가 없습니다. 다시 시도해 주세요.")
        return

    # 1) 기본 요금표
    base_fee = {
        1: 12180,
        2: 15830,
        3: 12180,
        4: 14610,
        5: 18170
    }[inputs["service_type"]]

    # 2) 아동 추가 요금 (기관연계 제외)
    child_fee = 0
    if inputs["service_type"] != 5:
        # 1명→0원, 2명→5295원, 3명→10590원
        child_fee = [0, 5295, 10590][inputs["children"] - 1]

    # 3) 야간/휴일 할증
    surcharge = 1.5 if (inputs["is_night"] or inputs["is_holiday"]) else 1.0

    # 4) 시간당 요금
    hourly_fee = int((base_fee + child_fee) * surcharge)
    total_fee  = int(hourly_fee * inputs["hours"])

    # 5) 본인부담율
    user_rate = [0.15, 0.4, 0.7, 0.85, 1.0][inputs["income_type"] - 1]
    user_fee  = total_fee * user_rate

    # 6) 다자녀 할인 적용
    if inputs["is_multi"] and inputs["income_type"] != 5:
        user_fee *= 0.9

    user_fee = int(user_fee)
    gov_fee  = total_fee - user_fee

    # — 출력
    st.markdown("<h3 style='text-align:center;'>🧾 요금 계산 결과</h3>", unsafe_allow_html=True)
    st.write(f"- 서비스 종류: **{inputs['service_type']}번**")
    st.write(f"- 이용 시간: **{inputs['hours']}시간**")
    st.write(f"- 동시 돌봄 아동 수: **{inputs['children']}명**")
    st.write(f"- 소득 유형: **{inputs['income_type']}번**")
    st.write(f"- 야간 이용: **{'예' if inputs['is_night'] else '아니오'}**, 휴일 이용: **{'예' if inputs['is_holiday'] else '아니오'}**")
    if inputs["income_type"] != 5:
        st.write(f"- 다자녀 가구: **{'예' if inputs['is_multi'] else '아니오'}**")
    st.markdown(f"---\n**총 이용요금:** {total_fee:,}원  \n(시간당 {hourly_fee:,}원 × {inputs['hours']}시간)")
    st.write(f"▪ 본인 부담금: {user_fee:,}원")
    st.write(f"▪ 정부 지원금: {gov_fee:,}원")

    if st.button("처음으로 돌아가기"):
        st.session_state.page = "home"
        st.rerun()
def page_caregiver_list():
    
    st.markdown("<h2 style='text-align: center;'>전체 돌보미 프로필</h2>", unsafe_allow_html=True)

    try:
        filter_age = st.selectbox(" 돌봄 가능 연령 필터", ["전체", "0~2세", "3~5세", "6세 이상", "전 연령"])
        filter_day = st.selectbox(" 활동 가능 요일 필터", ["전체", "월", "화", "수", "목", "금", "토", "일"])

        conn = sqlite3.connect("caregivers.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name, age, personality, career, philosophy, certification,
                   age_range, intro, phone, image_url, availability
            FROM caregivers
        """)
        caregivers = cursor.fetchall()
        conn.close()

        # 필터 적용 함수
        def matches(cg):
            age_range = cg[6]
            availability_text = cg[10].replace(" ", "")
            available_days = expand_days(availability_text)

            age_match = (
                filter_age == "전체" or
                (filter_age == "0~2세" and any(a in age_range for a in ["0", "1", "2"])) or
                (filter_age == "3~5세" and any(a in age_range for a in ["3", "4", "5", "초등"])) or
                (filter_age == "6세 이상" and any(a in age_range for a in ["6", "초등"])) or
                (filter_age == "전 연령" and "전" in age_range)
            )

            day_match = filter_day == "전체" or filter_day in available_days

            return age_match and day_match

        caregivers = [c for c in caregivers if matches(c)]

        if not caregivers:
            st.warning("조건에 맞는 돌보미가 없습니다. 필터 조건을 변경해보세요!")
            st.markdown("""
                <div style='text-align:center; color:#888;'>
                    <img src='https://cdn-icons-png.flaticon.com/512/4076/4076549.png' width='100'/>
                    <p style='margin-top:10px;'>조건에 해당하는 돌보미를 찾지 못했어요 😢<br>
                    필터를 변경하거나 전체 목록을 다시 확인해보세요.</p>
                </div>
            """, unsafe_allow_html=True)
            return

        items_per_page = 5
        total_pages = math.ceil(len(caregivers) / items_per_page)
        if "caregiver_page" not in st.session_state:
            st.session_state.caregiver_page = 1
        current_page = st.session_state.caregiver_page
        start_idx = (current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_items = caregivers[start_idx:end_idx]

        colors = ["#7993c1", "#ffb347", "#77dd77", "#f49ac2", "#aec6cf", "#fdfd96", "#ff6961"]
        for i, c in enumerate(current_items):
            color = colors[i % len(colors)]
            st.markdown(f"""
            <div style="background:#fff; border-left:8px solid {color};
                        border-radius:15px; box-shadow:0 4px 12px rgba(0,0,0,0.1);
                        padding:20px; margin:20px 0;">
                <div style="display:flex; align-items:center;">
                    <img src="{c[9]}" style="width:80px; height:80px; border-radius:50%; margin-right:20px;">
                    <div>
                        <h4> <strong>{c[0]}</strong> <span style='font-size:14px;'>(만 {c[1]}세)</span></h4>
                        <p><strong> 성격:</strong> {c[2]}</p>
                        <p><strong> 경력:</strong> {c[3]}</p>
                        <p><strong> 육아 가치관:</strong> {c[4]}</p>
                        <p><strong> 자격증:</strong> {c[5]}</p>
                        <p><strong> 돌봄 가능 연령:</strong> {c[6]}</p>
                        <p><strong> 활동 가능 시간:</strong> {c[10]}</p>
                        <p><strong> 전화번호:</strong> {c[8]}</p>
                    </div>
                </div>
                <p><strong>✍️ 자기소개:</strong></p>
                <blockquote style="margin-left:10px; font-style:italic; color:#444;">{c[7]}</blockquote>
            </div>
            """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if current_page > 1 and st.button("◀ 이전"):
                st.session_state.caregiver_page -= 1
                st.rerun()
        with col3:
            if current_page < total_pages and st.button("다음 ▶"):
                st.session_state.caregiver_page += 1
                st.rerun()

        st.markdown(f"<p style='text-align:center; color:gray;'>페이지 {current_page} / {total_pages}</p>", unsafe_allow_html=True)

        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        if st.button("🏠 홈으로 돌아가기"):
            st.session_state.page = "home"
            st.session_state.caregiver_page = 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
 
# ───────────────────────────────────────────────
# 돌보미 조건 설정 페이지
# ───────────────────────────────────────────────
def page_caregiver_conditions():
    st.subheader("🗓️ 돌보미 조건 설정")
    st.markdown("돌봄이 가능한 요일, 시간, 조건을 선택해주세요.")

    if "user_email" not in st.session_state:
        st.error("먼저 로그인해주세요.")
        return

    # ───── 요일 선택 ─────
    days = ["월", "화", "수", "목", "금", "토", "일"]
    selected_days = []
    select_all = st.checkbox("모든 요일 선택")
    cols = st.columns(7)
    for i, day in enumerate(days):
        if cols[i].checkbox(day, value=select_all, key=f"day_{day}"):
            selected_days.append(day)

    # ───── 시간대 추가 ─────
    st.markdown("<h4 style='color: #2c3e50;'>시간대 설정</h4>", unsafe_allow_html=True)
    if "edit_time_slots" not in st.session_state:
        st.session_state.edit_time_slots = []

    if st.button("⏰ 시간대 추가"):
        st.session_state.edit_time_slots.append({"start": 1, "end": 1})

    for i, slot in enumerate(st.session_state.edit_time_slots):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            slot["start"] = st.selectbox("시작 시간", range(1, 25), index=slot["start"]-1, key=f"start_{i}")
        with col2:
            slot["end"] = st.selectbox("종료 시간", range(1, 25), index=slot["end"]-1, key=f"end_{i}")
        with col3:
            if st.button("🗑️", key=f"delete_{i}"):
                st.session_state.edit_time_slots.pop(i)
                st.rerun()

    # ───── 특수아동 여부 ─────
    st.markdown("<h4 style='color: #2c3e50;'>특수아동 수용 여부</h4>", unsafe_allow_html=True)
    special_child = st.radio("", ["O", "X"], horizontal=True)

    # ───── 연령대 설정 ─────
    st.markdown("<h4 style='color: #2c3e50;'>수용 가능 연령대</h4>", unsafe_allow_html=True)
    age_range = st.slider("연령 범위 (단위: 세)", 0.25, 12.0, (1.0, 10.0), step=0.25, format="%.2f")
    col1, col2 = st.columns([5, 1])
    with col1:
        if st.button("돌아가기"):
            st.session_state.page = "start"  # 홈 화면을 표시하도록 페이지 상태 변경
            st.rerun()

    with col2:
        # ───── 저장 버튼 ─────
        if st.button("저장"):
            if not selected_days:
                st.warning("하나 이상의 요일을 선택해주세요.")
                st.stop()
            if not st.session_state.edit_time_slots:
                st.warning("하나 이상의 시간대를 추가해주세요.")
                st.stop()

            update_payload = {
                "email": st.session_state.user_email,
                "available_days": selected_days,
                "available_times": st.session_state.edit_time_slots,
                "special_child": special_child == "O",
                "age_min": age_range[0],
                "age_max": age_range[1]
            }
            try:
                res = requests.post("http://localhost:8005/caregiver/update-conditions", json=update_payload)
                res.raise_for_status()
                st.success("조건이 성공적으로 저장되었습니다.")
            except requests.exceptions.RequestException as e:
                st.error(f"저장 실패: {e}")



# 자동 로그인 후 시작 페이지에서 바로 home으로
# 자동 로그인 후 시작 → home 자동 전환
if st.session_state.get("logged_in") and st.session_state.page == "start":
    st.session_state.page = "home"
    st.rerun()

page = st.session_state.page

if page == "start":
    page_start()

elif page == "home":
    if st.session_state.user_role == "돌보미":
        page_caregiver_home()
    else:
        page_parent_home()

elif page == "recommend":
    page_recommend_service()

elif page == "recommend_result":
    page_recommend_result()

elif page == "chat":
    page_chat_talk()

elif page == "pricing":
    page_pricing()

elif page == "fee_result":
    page_fee_result()

elif page == "caregiver_conditions":
    page_caregiver_conditions()

elif page == "caregivers":
    page_caregiver_list()
    
elif page=="caregiver_personality":
    page_caregiver_personality()

# 더 만들 페이지가 생기면 여기 아래에 elif 로 추가
