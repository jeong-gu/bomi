import streamlit as st
import requests
from streamlit_lottie import st_lottie
import sqlite3
import math
import re
import time
import sqlite3, math
import streamlit as st
from datetime import datetime 

import requests

# ➕ 감정 점수 & 기록용 세션 상태 초기화
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = {"joy":0,"positive":0,"surprise":0,"anger":0,"sadness":0,"fear":0}
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []  # 슬픔 추이만 담을 경우




# 1) 전역 설정 & 세션 상태 초기화
############################################
st.set_page_config(page_title="여성가족부 JBNU 챗봇", page_icon="☁️", layout="centered")

RAG_API_URL = "http://localhost:8005/ask/"

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

# /* 페이지 전체 배경 컨테이너 */
# .block-container {
#   background-color: rgba(255, 255, 255, 0.85);
#   backdrop-filter: blur(10px);
#   padding: 2rem;
#   border-radius: 12px;
#   box-shadow: 0 4px 12px rgba(0,0,0,0.1);
# }
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
    background-color: #7993C1;
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
    """돌보미 성향 자가진단 챗 인터페이스"""
    if "caregiver_self_messages" not in st.session_state:
        st.session_state.caregiver_self_messages = [{
            "role": "assistant",
            "content": (
                "안녕하세요 😊 어떤 돌보미이신가요?\n"
                "예: '꼼꼼하게 약속을 지키는 편이에요', '아이의 감정에 귀 기울여요'"
            )
        }]
    if "last_caregiver_self_input" not in st.session_state:
        st.session_state.last_caregiver_self_input = None
    if "waiting_for_trait_response" not in st.session_state:
        st.session_state.waiting_for_trait_response = False

    st.markdown("<h3 style='text-align:center;'>📝 돌보미 성향 자가진단</h3>", unsafe_allow_html=True)
    col_text, col_save = st.columns([5,1])
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
                        traits = res1.json().get("traits", {})

                        # 누락된 항목 보완 (모든 trait이 빠졌을 경우도 대응)
                        default_traits = {
                            "diligent": 0.1,
                            "sociable": 0.1,
                            "cheerful": 0.1,
                            "warm": 0.1,
                            "positive": 0.1,
                            "observant": 0.1
                        }
                        for k, v in default_traits.items():
                            traits[k] = traits.get(k, v)

                        res2 = requests.post(
                            "http://localhost:8005/caregiver/update-traits",
                            json={"email": st.session_state.user_email, **traits}
                        )
                        res2.raise_for_status()

                        st.success("성향 점수가 성공적으로 저장되었어요! 🎉\n홈 화면으로 이동합니다.")
                        st.session_state.page = "start"
                        st.rerun()

                    except requests.exceptions.RequestException as e:
                        st.error(f"서버 요청 중 오류 발생: {e}")
                    except Exception as e:
                        st.error(f"예기치 않은 오류: {e}")




    # — 챗 기록 렌더링 —
    html = '<div class="chat-container">'
    if st.session_state.waiting_for_trait_response:
        html += '<div class="loading-bubble">답변 생성 중...</div>'
    for msg in st.session_state.caregiver_self_messages:
        cls = "user-bubble" if msg["role"]=="user" else "assistant-bubble"
        tag = "Q:" if msg["role"]=="user" else "A:"
        html += f'<div class="{cls}"><strong>{tag}</strong> {msg["content"]}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # — 입력창 —
    def _on_enter():
        ui = st.session_state.caregiver_self_input
        if not ui or ui == st.session_state.last_caregiver_self_input:
            return
        st.session_state.caregiver_self_messages.append({"role":"user","content":ui})
        st.session_state.last_caregiver_self_input = ui
        st.session_state.waiting_for_trait_response = True
        st.session_state.caregiver_self_input = ""

    st.text_input("", key="caregiver_self_input",
                  placeholder="성향에 대해 말씀해주세요!",
                  on_change=_on_enter)



    # — GPT 호출 및 답변 표시 —
    if st.session_state.waiting_for_trait_response:
        with st.spinner("답변 생성 중..."):
            resp = requests.post(
                RAG_API_URL,
                json={"prompt": st.session_state.last_caregiver_self_input,
                      "category":"caregiver_personality"}
            )
        answer = resp.json().get("answer", "응답이 없습니다.")
        st.session_state.caregiver_self_messages.append(
            {"role":"assistant","content":answer}
        )
        st.session_state.waiting_for_trait_response = False
        st.rerun()









############################################
# 4) 페이지별 함수
############################################
def page_start():
    import streamlit as st
    import requests

    st.markdown("<h2 style='text-align:center;'> 로그인 또는 회원가입</h2>", unsafe_allow_html=True)
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
                <h4 style='color: #2c3e50;'>돌봄 가능 연령대</h4>
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






############################################







# # 3) Lottie 애니메이션 로딩(옵션)
# ############################################
# def load_lottie_url(url: str):
#     try:
#         r = requests.get(url)
#         if r.status_code == 200:
#             return r.json()
#     except:
#         pass
#     return None

# lottie_welcome_url = "https://assets10.lottiefiles.com/packages/lf20_5e7wgehs.json"
# lottie_welcome = load_lottie_url(lottie_welcome_url)
# if "trigger_rerun" not in st.session_state:
#     st.session_state.trigger_rerun = False
# ############################################


########################################
# # 📝 타이핑 효과 함수 (중앙 정렬 유지)
# def typewriter_effect(text, key, delay=0.1):
#     """한 글자씩 출력하는 효과 (중앙 정렬 유지)"""
#     container = st.empty()
#     displayed_text = ""

#     for char in text:
#         displayed_text += char
#         container.markdown(
#             f"<h3 style='text-align: center;'>{displayed_text}</h3>", unsafe_allow_html=True
#         )
#         time.sleep(delay)
# ########################################


# ########################################
# def page_userinfo():
#     # 🎈 페이지 진입 시 풍선 효과
#     st.balloons()

#     # 🎨 페이지 스타일 설정
#     st.markdown("""
#         <style>
#             /* 전체 컨테이너 높이 조정 */
#             .block-container {
#                 min-height: 100vh;  /* 전체 화면 높이 */
#                 display: flex;
#                 flex-direction: column;
#                 justify-content: center;
#                 align-items: center;
#             }

#             /* 제목과 부제목 스타일 */
#             .title-container {
#                 text-align: center;
#                 font-weight: bold;
#                 font-size: 40px;
#                 margin-bottom: 10px;
#                 color: #7993c1;
#             }
#             .subtitle {
#                 text-align: center;
#                 font-size: 11px;
#                 color: #7993c1;
#             }

#             /* 안내 문구 (아래에 입력 후 Enter를 눌러주세요) 중앙 정렬 */
#             .input-guide {
#                 text-align: center;
#                 font-size: 12px;
#                 font-weight: basic;
#                 color: #7993c1;
#                 margin-bottom: 5px;
#             }
#         </style>
#     """, unsafe_allow_html=True)

#     # 🏡 화면 중앙 정렬 텍스트 (타이핑 효과)
#     typewriter_effect(" 만나서 반가워요!", key="title", delay=0.07)
#     time.sleep(0.5)  # 첫 번째 문장 출력 후 살짝 대기
#     typewriter_effect("이름을 알려주세요!", key="subtitle", delay=0.07)

#     # 🌥️ 로딩 애니메이션 or 이미지
#     if lottie_welcome:
#         st_lottie(lottie_welcome, height=250, key="welcome_lottie")
#     else:
#         st.image("https://via.placeholder.com/200x100?text=Loading+Clouds", use_container_width=True)

#     # 📝 안내 문구 중앙 정렬
#     st.markdown("<p class='input-guide'>아래에 입력 후 <b>Enter</b>를 눌러주세요</p>", unsafe_allow_html=True)

#     # 👤 이름 입력 필드
#     def on_name_submit():
#         if st.session_state.name:
#             st.session_state.user_name = st.session_state.name
#             st.session_state.page = "home"
#             st.session_state.trigger_rerun = True

#     st.text_input("", key="name", on_change=on_name_submit, placeholder="이름을 입력해주세요")  # 입력창은 그대로 유지

#     # 🔄 trigger_rerun 체크 후 페이지 리로드
#     if st.session_state.trigger_rerun:
#         st.session_state.trigger_rerun = False
#         st.rerun()
# ########################################



# ########################################
# def page_home():
#     # ✅ 로그인 확인
#     if not st.session_state.get("logged_in"):
#         st.warning("로그인이 필요합니다.")
#         st.session_state.page = "start"
#         st.rerun()

#     # ✅ 사용자 이름 초기화
#     if "user_name" not in st.session_state:
#         st.session_state.user_name = "사용자"  # 기본값

#     user_name = st.session_state.user_name

#     # 🌟 환영 메시지 중앙 정렬
#     st.markdown(f"""
#     <h3 style="text-align: center;">　환영해요, <strong>{user_name}</strong>님.</h3>
#     <p style="text-align: center;">무엇을 해볼까요?</p>
#     """, unsafe_allow_html=True)

#     # ✅ CSS 스타일
#     st.markdown("""
#         <style>
#         .block-container {
#                 min-height: 100vh;
#                 display: flex;
#                 flex-direction: column;
#                 justify-content: center;
#                 align-items: center;
#         }
#         .stButton > button {
#             display: flex;
#             flex-direction: column;
#             justify-content: center;
#             align-items: center;
#             width: 100%;
#             height: 120px;
#             padding: 10px;
#             font-size: 40px;
#             font-weight: bold;
#             text-align: center;
#             background: white;
#             border: 3px solid #7993c1;
#             border-radius: 25px;
#             color: #2c3e50;
#             cursor: pointer;
#             transition: all 0.3s ease-in-out;
#             box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
#             white-space: pre-line;
#         }
#         .stButton > button:hover {
#             background: #7993c1;
#             color: white;
#             box-shadow: 2px 2px 10px rgba(74, 111, 165, 0.5);
#         }
#         .stButton > button:active {
#             background: #7993c1;
#             color: white;
#             box-shadow: 2px 2px 10px rgba(74, 111, 165, 0.8);
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     # ✅ 버튼 3개 배치
#     col1, col2, col3, col4 = st.columns(4)
            
#     with col1:
#         if st.button("✮⋆\n정보용\n.", key="chat_btn"):
#             st.session_state.page = "chat"
#             st.rerun()

#     with col2:
#         if st.button("🎯\n추천용\n.", key="recommend_btn"):
#             # 통합 챗봇으로 바로 이동
#             st.session_state.page = "recommend"
#             # 대화 관련 상태 초기화
#             st.session_state.recommend_messages = []
#             st.session_state.last_recommend_input = None
#             st.session_state.waiting_for_recommend_response = False
#             st.session_state.recommend_done = False
#             st.session_state.recommendations = []
#             st.rerun()

#     with col3:
#         if st.button("📊\n요금\n산정", key="pricing_btn"):
#             st.session_state.page = "pricing"
#             st.rerun()
            
#     with col4:
#         if st.button("👩‍🍼\n돌보미 목록", key="caregivers_btn"):
#             st.session_state.page = "caregivers"
#             st.rerun()            
########################################*





##################################################
##################################################
# ───────────────────────────────────────────────────
# 돌보미 전용 홈 페이지
# ───────────────────────────────────────────────────
def page_caregiver_home():
    
    # 1) 로그인 확인
    if not st.session_state.get("logged_in"):
        st.warning("로그인이 필요합니다.")
        st.session_state.page = "start"
        st.rerun()

    # 2) 사용자 이름
    user_name = st.session_state.get("user_name", "사용자")

    # 3) 환영 메시지
    st.markdown(f"""
    <h3 style="text-align: center;">환영해요, <strong>{user_name}</strong> 돌보미님.</h3>
    <p style="text-align: center;">무엇을 해보실까요?</p>
    """, unsafe_allow_html=True)





    # 4) CSS (기존 그대로)
    st.markdown("""
        <style>
        .block-container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        /* 페이지 전체 배경 컨테이너 */
        .block-container {
            background-color: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
        .stButton > button {
            background-color: #7993C1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            width: 110px;
            height: 110px;
            padding: 10px;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            background: white;
            border: 3px solid #7993c1;
            border-radius: 25px;
            color: #2c3e50;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
            white-space: pre-line;
        }
        .stButton > button:hover {
            background: #7993c1;
            color: white;
            box-shadow: 2px 2px 10px rgba(74, 111, 165, 0.5);
        }
        .stButton > button:active {
            background: #7993c1;
            color: white;
            box-shadow: 2px 2px 10px rgba(74, 111, 165, 0.8);
        }
        </style>
    """, unsafe_allow_html=True)

    # 5) 메뉴 버튼 5개 배치
    cols = st.columns(5)
    with cols[0]:
        if st.button("✮⋆\n정보용", key="care_info"):
            st.session_state.page = "chat"
            st.rerun()
    with cols[1]:
        if st.button("🎯\n성향분석", key="care_recommend"):
            st.session_state.page = "recommend"
            st.rerun()
    with cols[2]:
        if st.button("📊\n요금산정", key="care_pricing"):
            st.session_state.page = "pricing"
            st.rerun()
    with cols[3]:
        if st.button("👩‍🍼\n돌보미목록", key="care_list"):
            st.session_state.page = "caregivers"
            st.rerun()
    with cols[4]:
        if st.button("⚙️\n조건설정", key="care_settings"):
            st.session_state.page = "caregiver_settings"
            st.rerun()





# # ───────────────────────────────────────────────────
# # 부모(고객) 전용 홈 페이지
# # ───────────────────────────────────────────────────
def page_parent_home():
    # 1) 로그인 확인
    if not st.session_state.get("logged_in"):
        st.warning("로그인이 필요합니다.")
        st.session_state.page = "start"
        st.rerun()

    # 2) 사용자 이름
    if "user_name" not in st.session_state:
        st.session_state.user_name = "사용자"
    user_name = st.session_state.user_name

    # 3) 환영 메시지
    st.markdown(f"""
    <h3 style="text-align: center;">　환영해요, <strong>{user_name}</strong> 부모님.</h3>
    <p style="text-align: center;">무엇을 해보실까요?</p>
    """, unsafe_allow_html=True)

    # 4) CSS (기존 그대로)
    st.markdown("""
        <style>
        .block-container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        /* 페이지 전체 배경 컨테이너 */
        .block-container {
            background-color: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
        .stButton > button {
            background-color: #7993C1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            width: 110px;
            height: 110px;
            padding: 10px;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            background: white;
            border: 3px solid #7993c1;
            border-radius: 25px;
            color: #2c3e50;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
            white-space: pre-line;
        }
        .stButton > button:hover {
            background: #7993c1;
            color: white;
            box-shadow: 2px 2px 10px rgba(74, 111, 165, 0.5);
        }
        .stButton > button:active {
            background: #7993c1;
            color: white;
            box-shadow: 2px 2px 10px rgba(74, 111, 165, 0.8);
        }
        </style>
    """, unsafe_allow_html=True)

    # 5) 버튼 4개 배치
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("✮\n정보용 챗봇", key="parent_chat_btn"):
            st.session_state.page = "chat"; st.rerun()
    with col2:
        if st.button("🎯\n추천용 챗봇", key="parent_recommend_btn"):
            st.session_state.page = "recommend"; st.rerun()
    with col3:
        if st.button("📊\n요금산정", key="parent_pricing_btn"):
            st.session_state.page = "pricing"; st.rerun()
    with col4:
        if st.button("👩‍🍼\n돌보미목록", key="parent_list_btn"):
            st.session_state.page = "caregivers"; st.rerun()

# ##################################################
# ##################################################


# ——————————————————————————————————————————————————
# (1) 전역에 한번만 CSS 추가
# ——————————————————————————————————————————————————
# 전역에 한번만 선언한 CSS 수정
# 2) CSS 스타일
############################################



# ───────────────────────────────────────────────────
# # 돌보미 전용 홈 페이지
# # ───────────────────────────────────────────────────
# def page_caregiver_home():
#     # 1) 로그인 확인
#     if not st.session_state.get("logged_in"):
#         st.warning("로그인이 필요합니다.")
#         st.session_state.page = "start"
#         st.rerun()

#     # 2) 사용자 이름
#     user_name = st.session_state.get("user_name", "사용자")

#     # 3) 상단: 환영 메시지
#     st.markdown(
#         f"<h3 style='text-align: left;'>환영해요, <strong>{user_name}</strong> 돌보미님.</h3>",
#         unsafe_allow_html=True
#     )

#     # 4) 기존 CSS 유지
#     st.markdown("""
#         <style>
#         .block-container {
#             min-height: 100vh; display:flex; flex-direction:column;
#             justify-content:center; align-items:center;
#         }
#         .stButton > button {
#             display:flex; flex-direction:column; justify-content:center;
#             align-items:center; width:100%; height:120px; padding:10px;
#             font-size:40px; font-weight:bold; text-align:center;
#             background:white; border:3px solid #7993c1; border-radius:25px;
#             color:#2c3e50; cursor:pointer; transition:all .3s ease-in-out;
#             box-shadow:2px 2px 8px rgba(0,0,0,0.1); white-space:pre-line;
#         }
#         .stButton > button:hover {
#             background:#7993c1; color:white;
#             box-shadow:2px 2px 10px rgba(74,111,165,0.5);
#         }
#         .stButton > button:active {
#             background:#7993c1; color:white;
#             box-shadow:2px 2px 10px rgba(74,111,165,0.8);
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     # 5) 메뉴 버튼 + 조건 설정
#     cols = st.columns(5)
#     if cols[0].button("정보용", key="care_info"):
#         st.session_state.page = "chat"; st.rerun()
#     if cols[1].button("성향분석", key="care_recommend"):
#         st.session_state.page = "recommend"; st.rerun()
#     if cols[2].button("요금산정", key="care_pricing"):
#         st.session_state.page = "pricing"; st.rerun()
#     if cols[3].button("돌보미목록", key="care_list"):
#         st.session_state.page = "caregivers"; st.rerun()
#     if cols[4].button("조건설정", key="care_settings"):
#         st.session_state.page = "caregiver_settings"; st.rerun()

#     # 6) 하단 고정 로그아웃 버튼
#     st.markdown('<div class="bottom-logout">', unsafe_allow_html=True)
#     if st.button("로그아웃", key="logout_bottom"):
#         for k in ["logged_in","user_email","user_role","user_name"]:
#             st.session_state.pop(k, None)
#         st.session_state.page = "start"
#         st.rerun()
#     st.markdown('</div>', unsafe_allow_html=True)


# # ───────────────────────────────────────────────────
# # 부모(고객) 전용 홈 페이지
# # # ───────────────────────────────────────────────────
# def page_parent_home():
#     # 1) 로그인 확인
#     if not st.session_state.get("logged_in"):
#         st.warning("로그인이 필요합니다.")
#         st.session_state.page = "start"
#         st.rerun()

#     # 2) 사용자 이름
#     user_name = st.session_state.get("user_name", "사용자")

#     # 3) 상단: 환영 메시지
#     st.markdown(
#         f"<h3 style='text-align: left;'>환영해요, <strong>{user_name}</strong> 부모님.</h3>",
#         unsafe_allow_html=True
#     )

#     # 4) CSS (기존 그대로)
#     st.markdown("""
#         <style>
#         .block-container {
#             min-height: 100vh; display:flex; flex-direction:column;
#             justify-content:center; align-items:center;
#         }
#         .stButton > button {
#             display:flex; flex-direction:column; justify-content:center;
#             align-items:center; width:100%; height:120px; padding:10px;
#             font-size:40px; font-weight:bold; text-align:center;
#             background:white; border:3px solid #7993c1; border-radius:25px;
#             color:#2c3e50; cursor:pointer; transition:all .3s ease-in-out;
#             box-shadow:2px 2px 8px rgba(0,0,0,0.1); white-space:pre-line;
#         }
#         .stButton > button:hover {
#             background:#7993c1; color:white;
#             box-shadow:2px 2px 10px rgba(74,111,165,0.5);
#         }
#         .stButton > button:active {
#             background:#7993c1; color:white;
#             box-shadow:2px 2px 10px rgba(74,111,165,0.8);
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     # 5) 메뉴 버튼 (조건설정 제외)
#     cols = st.columns(4)
#     if cols[0].button("정보용", key="parent_info"):
#         st.session_state.page = "chat"; st.rerun()
#     if cols[1].button("추천용", key="parent_recommend"):
#         st.session_state.page = "recommend"; st.rerun()
#     if cols[2].button("요금산정", key="parent_pricing"):
#         st.session_state.page = "pricing"; st.rerun()
#     if cols[3].button("돌보미목록", key="parent_list"):
#         st.session_state.page = "caregivers"; st.rerun()
# #

# #
# def page_caregiver_home():
#     # 1) 로그인 확인
#     if not st.session_state.get("logged_in"):
#         st.warning("로그인이 필요합니다.")
#         st.session_state.page = "start"
#         st.rerun()

#     user_name = st.session_state.get("user_name", "사용자")

#     # ────────────────────────────────────────────────
#     # 이 함수 전용 CSS
#     # ────────────────────────────────────────────────
#     st.markdown("""
#     <style>
#       /* 페이지 전체를 꽉 채우는 컨테이너 */
#       .block-container {
#         min-height: 100vh;
#         display: flex;
#         flex-direction: column;
#         justify-content: flex-start;
#         align-items: center;
#       }
#       /* 헤더 로그아웃 버튼 스타일 */
#       .logout-btn > button {
#         background: transparent !important;
#         border: none !important;
#         font-size: 1.5rem !important;
#         cursor: pointer !important;
#       }
#       .logout-btn > button:hover {
#         color: #ff6961 !important;
#       }
#       /* 메뉴 버튼 고정 크기 + 간격 */
#       .menu-btns .stButton > button {
#         width: 140px !important;
#         margin: 0 10px !important;
#         white-space: nowrap !important;
#       }
#     </style>
#     """, unsafe_allow_html=True)

#     # ────────────────────────────────────────────────
#     # 2) 헤더: 로그아웃 버튼 + 환영 문구
#     # ────────────────────────────────────────────────
#     st.markdown(
#         '<div class="logout-btn" style="width:100%; text-align:right; padding:10px 0;">',
#         unsafe_allow_html=True
#     )
#     if st.button("⏻", key="logout_caregiver"):
#         for k in ["logged_in","user_email","user_role","user_name"]:
#             st.session_state.pop(k, None)
#         st.session_state.page = "start"
#         st.rerun()
#     st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown(
#         f"<h3 style='width:100%; text-align:left; padding-bottom:20px;'>"
#         f"환영해요, <strong>{user_name}</strong> 돌보미님.</h3>",
#         unsafe_allow_html=True
#     )

#     # ────────────────────────────────────────────────
#     # 3) 메뉴 버튼 2행 배치 (가운데 대칭)
#     # ────────────────────────────────────────────────
#     # 래퍼 div 에 클래스를 걸어 버튼 스타일 적용
#     st.markdown('<div class="menu-btns" style="width:100%;">', unsafe_allow_html=True)

#     # --- 1행: 3개 버튼 (정보용 / 성향분석 / 요금산정)
#     row1 = st.columns(5)
#     if row1[1].button("정보용"):
#         st.session_state.page = "chat"; st.rerun()
#     if row1[2].button("성향분석"):
#         st.session_state.page = "recommend"; st.rerun()
#     if row1[3].button("요금산정"):
#         st.session_state.page = "pricing"; st.rerun()

#     # --- 2행: 2개 버튼 (돌보미목록 / 조건설정)
#     row2 = st.columns(5)
#     if row2[1].button("돌보미목록"):
#         st.session_state.page = "caregivers"; st.rerun()
#     if row2[3].button("조건설정"):
#         st.session_state.page = "caregiver_settings"; st.rerun()

#     st.markdown('</div>', unsafe_allow_html=True)
# #




# # !!!!!!!11
# def page_parent_home():
#     # 로그인 확인
#     if not st.session_state.get("logged_in"):
#         st.warning("로그인이 필요합니다.")
#         st.session_state.page = "start"
#         st.rerun()

#     user_name = st.session_state.get("user_name", "사용자")

#     # ── 헤더: 로그아웃 버튼 + 환영 메시지
#     st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
#     if st.button("⏻", key="logout_parent"):
#         for k in ["logged_in","user_email","user_role","user_name"]:
#             st.session_state.pop(k, None)
#         st.session_state.page = "start"
#         st.rerun()
#     st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown(
#         f"<h3 style='text-align:left;'>환영해요, <strong>{user_name}</strong> 부모님.</h3>",
#         unsafe_allow_html=True
#     )

#     # ── 메뉴 버튼 (4개)
#     c1, c2, c3, c4 = st.columns(4)
#     if c1.button("정보용"):      st.session_state.page="chat";       st.rerun()
#     if c2.button("추천용"):      st.session_state.page="recommend";  st.rerun()
#     if c3.button("요금산정"):    st.session_state.page="pricing";    st.rerun()
#     if c4.button("돌보미목록"):  st.session_state.page="caregivers"; st.rerun()
# #

########################################
def page_chat_talk():
    # ✅ 상태 변수 설정
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # 수다 대화 상태 변수 추가
    if "last_chat_input" not in st.session_state:
        st.session_state.last_chat_input = None  
    if "waiting_for_chat_response" not in st.session_state:
        st.session_state.waiting_for_chat_response = False

    # ✅ **CSS 스타일 수정 (입력창을 채팅창 내부에 완전히 포함)**
    st.markdown("""
    <style>
/* 페이지 전체 배경 컨테이너 */
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
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
    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        if st.button("◀", key="back_chat_btn"):
            st.session_state.page = "home"
            st.rerun()

    with col2:
        st.markdown(f"<h3 style='text-align: center;'>무엇이 궁금하신가요?</h3>", unsafe_allow_html=True)

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
                    json={"prompt": st.session_state.chat_messages[-1]["content"]}
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
#
def page_pricing():
    # ✅ 상단 타이틀 바 (버튼 위치 조정)
    st.markdown("""<div style='margin-top:30px;'></div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 10, 1])
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

    # ✅ 전체 레이아웃 스타일 수정
    st.markdown("""
    <style>
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        padding: 3rem 2rem 5rem 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

    # ─────────── 설문 입력 폼 ───────────
    SERVICE_OPTIONS = [
        (1, "시간제 기본형"),
        (2, "시간제 종합형"),
        (3, "영아종일제"),
        (4, "질병감염아동지원"),
        (5, "기관연계서비스"),
    ]

    with st.form("fee_form"):
        sel = st.selectbox("서비스 종류 선택", options=SERVICE_OPTIONS,
                           format_func=lambda x: f"{x[0]}: {x[1]}", key="service_type")
        service_type = sel[0]

        hours = st.number_input("이용 시간(시간 단위, 예: 3.5)",
                                min_value=0.5, max_value=24.0, value=1.0, step=0.5, key="hours")

        max_children = 5 if service_type == 5 else 3
        num_children = st.number_input(
            f"동시 돌봄 아동 수 (1-{max_children})",
            min_value=1, max_value=max_children, value=1, step=1, key="children"
        )

        INCOME_OPTIONS = [
            (1, "가형 (중위소득 75% 이하)"),
            (2, "나형 (중위소득 120% 이하)"),
            (3, "다형 (중위소득 150% 이하)"),
            (4, "라형 (중위소득 200% 이하)"),
            (5, "마형 (중위소득 200% 초과)"),
        ]
        sel_inc = st.selectbox("소득 유형 선택", options=INCOME_OPTIONS,
                               format_func=lambda x: f"{x[0]}: {x[1]}", key="income_type")
        income_type = sel_inc[0]

        is_night = st.checkbox("야간 시간대 이용 (22시~06시)", key="is_night")
        is_holiday = st.checkbox("휴일(일요일/공휴일) 이용", key="is_holiday")

        is_multi_child = False
        if income_type != 5:
            is_multi_child = st.checkbox("다자녀 가구 여부 (2명 이상)", key="is_multi")

        submitted = st.form_submit_button("다음 → 요금 계산")
        if submitted:
            st.session_state.fee_inputs = {
                "service_type": service_type,
                "hours": hours,
                "children": num_children,
                "income_type": income_type,
                "is_night": is_night,
                "is_holiday": is_holiday,
                "is_multi": is_multi_child,
            }
            st.session_state.page = "fee_result"
            st.rerun()

########################################
########################################



########################################
# def page_chat_counseling():
#

def page_chat_counseling():
    import requests

    cat = st.session_state.selected_category or "고민"

    # ───────────────────────────────────────────────────
    # 1) 상태 초기화
    # ───────────────────────────────────────────────────
    if "last_input" not in st.session_state:
        st.session_state.last_input = None
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False
    if "counseling_done" not in st.session_state:
        st.session_state.counseling_done = False
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "current_emotion" not in st.session_state:
        st.session_state.current_emotion = {"joy":0,"positive":0,"surprise":0,"anger":0,"sadness":0,"fear":0}
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []

    # ───────────────────────────────────────────────────
    # 2) CSS & 헤더 (기존 그대로)
    # ───────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css?family=Poppins:300,400,600&display=swap');

    * {
      font-family: 'Poppins', sans-serif;
    }
    body {
      background: linear-gradient(135deg, #dbeeff 25%, #ffffff 100%) no-repeat center center fixed;
      background-size: cover;
    }
    .block-container {
      background-color: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(10px);
      padding: 2rem;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .user-bubble {
      text-align: left;
      background-color: #cceeff;
      padding: 10px;
      border-radius: 10px;
      margin-bottom: 10px;
      max-width: 70%;
      margin-left: auto;
      box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .assistant-bubble {
      text-align: left;
      background-color: #ffffff;
      padding: 10px;
      border-radius: 10px;
      margin-bottom: 10px;
      max-width: 70%;
      margin-right: auto;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-container {
      width: 90%;
      max-width: 600px;
      height: 70vh;
      display: flex;
      flex-direction: column-reverse;
      overflow-y: auto;
      padding: 15px;
      background: white;
      margin: auto;
      border-radius: 15px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
      position: relative;
    }
    .loading-bubble {
      background: #fff2c7;
      padding: 12px;
      border-radius: 20px;
      margin: 5px 0;
      max-width: 70%;
      margin-right: auto;
      text-align: left;
      box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
      font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        if st.button("◀", key="back_btn"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        user_name = st.session_state.get("user_name", "사용자")
        st.markdown(
            f"<h3 style='text-align: center;'>🤓 {user_name}님, 고민을 알려주세요</h3>",
            unsafe_allow_html=True
        )
    with col3:
        if st.button("🏠", key="home_btn"):
            st.session_state.page = "home"
            st.rerun()

    # ───────────────────────────────────────────────────
    # 3) 감정 기반 공감 배너
    # ───────────────────────────────────────────────────
    emo = st.session_state.current_emotion
    if emo.get("sadness", 0) > 0.5:
        st.warning("요즘 마음이 슬퍼 보이시네요. 편하게 이야기해 주세요. 🧸")
    elif emo.get("joy", 0) > 0.5:
        st.success("기분이 좋아 보이시네요! 어떤 이야기를 나눠볼까요? 😊")

    # ───────────────────────────────────────────────────
    # 4) 메시지 렌더링 (기존 그대로)
    # ───────────────────────────────────────────────────
    messages_html = '<div class="chat-container" id="chat-messages">'
    if st.session_state.waiting_for_response:
        messages_html += '<div class="loading-bubble">🐝 답변 생성 중...</div>'
    for msg in st.session_state.counseling_messages:
        cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        tag = "Q:" if msg["role"] == "user" else "A:"
        messages_html += f'<div class="{cls}"><strong>{tag}</strong> {msg["content"]}</div>'
    messages_html += '<div id="chat-end"></div></div>'
    messages_html += """
    <script>
        document.getElementById('chat-end')?.scrollIntoView({ behavior: 'smooth' });
    </script>
    """
    st.markdown(messages_html, unsafe_allow_html=True)

    # ───────────────────────────────────────────────────
    # 5) 사용자 입력 + 감정 콜
    # ───────────────────────────────────────────────────
    user_q = st.text_input("질문을 입력하세요!", key="chat_input", label_visibility="collapsed")
    if user_q and user_q != st.session_state.last_input:
        st.session_state.counseling_messages.append({"role": "user", "content": user_q})
        st.session_state.last_input = user_q
        st.session_state.waiting_for_response = True

        # ➡️ /emotion/ 호출
        emo = requests.post(
            "http://localhost:8005/emotion/",
            json={"prompt": user_q, "category": "general_chat"}
        ).json().get("scores", {})
        st.session_state.current_emotion = emo
        # 슬픔 추이 저장
        hist = st.session_state.emotion_history + [emo.get("sadness", 0)]
        st.session_state.emotion_history = hist[-10:]
        st.rerun()

    # ───────────────────────────────────────────────────
    # 6) LLM 응답 생성 (감정 힌트 포함)
    # ───────────────────────────────────────────────────
    if st.session_state.waiting_for_response:
        with st.spinner("🐝 답변 생성 중..."):
            lead = ""
            if emo.get("sadness", 0) > 0.5:
                lead = "[기분: 슬픔↑] "
            elif emo.get("anger", 0) > 0.5:
                lead = "[기분: 분노↑] "
            cat_clean = cat.replace("🏠 ","") \
                           .replace("💼 ","") \
                           .replace("💰 ","") \
                           .replace("🛡️ ","") \
                           .replace("📱 ","") \
                           .replace("🆘 ","")
            payload = {"prompt": lead + user_q, "category": cat_clean}
            try:
                resp = requests.post(RAG_API_URL, json=payload)
                resp.raise_for_status()
                answer = resp.json().get("answer", "🚨 응답 없음.")
            except Exception as e:
                answer = f"오류 발생: {e}"
        st.session_state.counseling_messages.append({"role": "assistant", "content": answer})
        st.session_state.waiting_for_response = False
        st.rerun()

    # ───────────────────────────────────────────────────
    # 7) 슬픔 추이 차트
    # ───────────────────────────────────────────────────
    st.markdown("**😢 슬픔 추이 (최근 10회)**")
    st.line_chart(st.session_state.emotion_history)

    # ───────────────────────────────────────────────────
    # 8) 자동 성향 저장 + 추천 기능 (감정 가중치 포함)
    # ───────────────────────────────────────────────────
    user_messages = [m["content"] for m in st.session_state.counseling_messages if m["role"] == "user"]
    if len(user_messages) >= 5 and not st.session_state.counseling_done:
        try:
            user_email = st.session_state.user_email
            with st.spinner("👀 부모님의 성향을 분석하고 돌보미를 추천 중입니다..."):
                # 성향 분석
                pref_resp = requests.post(
                    "http://localhost:8005/user/preference/from-chat",
                    json={"email": user_email, "history": user_messages}
                )
                pref_resp.raise_for_status()
                # 돌보미 추천 (감정 포함)
                rec_resp = requests.post(
                    "http://localhost:8005/recommend/caregiver",
                    json={"email": user_email, "emotion": st.session_state.current_emotion}
                )
                rec_resp.raise_for_status()
                st.session_state.recommendations = rec_resp.json().get("recommendations", [])
                st.session_state.counseling_done = True
        except Exception as e:
            st.session_state.counseling_done = True
            st.error(f"🔴 추천 실패: {type(e).__name__} - {e}")

    # ───────────────────────────────────────────────────
    # 9) 추천 돌보미 출력 (기존 그대로)
    # ───────────────────────────────────────────────────
    if st.session_state.recommendations:
        st.markdown("---")
        st.subheader("🧡 추천 돌보미 Top 3")
        for r in st.session_state.recommendations:
            st.markdown(f"""
**👩‍🍼 {r['name']}** (나이: {r['age']}세)  
📝 {r['personality']}  
💡 유사도: **{r['similarity']*100:.2f}%**
""")


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
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    .stButton > button {
        background-color: #7993C1;
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

import sqlite3
import math
import re

# 요일 구간을 실제 요일 리스트로 변환하는 함수

#
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
#

#
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
#

####################################### def page_caregiver_list():
import altair as alt
import json
import streamlit as st
import sqlite3
import math
import re
import pandas as pd
from datetime import datetime

# ────────────────────────────────────────────────────────────────
def expand_days(text: str):
    order = ["월", "화", "수", "목", "금", "토", "일"]
    s = set(); t = (text or "").replace(" ", "")
    if "주말" in t: s |= {"토", "일"}
    for a, b in re.findall(r"([월화수목금토일])~([월화수목금토일])", t):
        i, j = order.index(a), order.index(b)
        s |= set(order[i:j + 1] if i <= j else order[i:] + order[:j + 1])
    for d in order:
        if d in t: s.add(d)
    return s

def format_age(age_float: float) -> str:
    yrs = int(age_float)
    mons = round((age_float - yrs) * 12)
    if mons == 12:
        yrs += 1; mons = 0
    return f"{yrs}세 {mons}개월" if mons else f"{yrs}세"

def page_caregiver_list():

    # 사용자 목록 불러오기
    with sqlite3.connect("users.db", check_same_thread=False) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT c.id, u.username FROM caregivers AS c
            JOIN users AS u ON c.user_id = u.id
        """)
        all_rows = cur.fetchall()



    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        if st.button("◀", key="back_btn"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        user_name = st.session_state.get("user_name", "사용자")
        st.markdown("<h2 style='text-align:center;'>전체 돌보미 프로필</h2>", unsafe_allow_html=True)
    with col3:
        if st.button("🏠", key="home_btn"):
            st.session_state.page = "home"
            st.rerun()



    all_names = [r[1] for r in all_rows]
    sel = st.selectbox("돌보미 선택", all_names, key="selected_name")

    col1, col2, col3 = st.columns([2, 10, 2])

    with col1:
        if st.button("매칭하기", key="match_btn"):
            idx = all_names.index(sel)
            st.session_state.matched_id = all_rows[idx][0]
            st.session_state.matched_name = sel
            st.session_state.show_review_button = True
            st.session_state.show_review_input = False
            st.success("✅ 매칭 성공!")

    with col2:
        st.markdown("<h2 style='text-align:center;'></h2>", unsafe_allow_html=True)

    with col3:
        if st.session_state.get("show_review_button"):
            if st.button("후기 작성", key="review_btn"):
                st.session_state.show_review_input = True
                st.rerun()







    if st.session_state.get("show_review_input"):
        def add_review():
            txt = st.session_state.review_input.strip()
            if not txt: return
            with sqlite3.connect("users.db", isolation_level=None) as conn:
                conn.execute("""
                    INSERT INTO reviews (caregiver_id, parent_name, content, timestamp)
                    VALUES (?,?,?,?)
                """, (
                    st.session_state.matched_id,
                    st.session_state.get("user_name", "부모님"),
                    txt,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            st.session_state.review_input = ""
            st.success("✅ 후기 등록 완료!")

        st.text_input("후기를 입력하세요 (엔터로 제출)", key="review_input",
                      placeholder="엔터 키로 제출", on_change=add_review)

    # 필터
    filter_age = st.selectbox("돌봄 가능 아동 연령 필터", ["전체", "0~2세", "3~5세", "6세 이상", "전 연령"])
    filter_day = st.selectbox("활동 가능 요일 필터", ["전체", "월", "화", "수", "목", "금", "토", "일"])

    with sqlite3.connect("users.db", check_same_thread=False) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT c.id, u.username, c.age, u.phone,
                   c.available_days, c.available_times,
                   c.special_child, c.age_min, c.age_max,
                   c.diligent, c.sociable, c.cheerful,
                   c.warm, c.positive, c.observant
            FROM caregivers AS c
            JOIN users AS u ON c.user_id = u.id
        """)
        rows = cur.fetchall()

    def matches(row):
        _, _, _, _, days_str, _, _, amin, amax, *_ = row
        days = expand_days(days_str)
        if filter_day != "전체" and filter_day not in days:
            return False
        if filter_age == "0~2세": return amax <= 2
        if filter_age == "3~5세": return amin <= 5 and amax >= 3
        if filter_age == "6세 이상": return amin >= 6
        return True

    caregivers = [r for r in rows if matches(r)]
    if not caregivers:
        st.warning("조건에 맞는 돌보미가 없습니다.")
        return

    # 페이징
    per_page = 5
    total_pages = math.ceil(len(caregivers) / per_page)
    page = st.session_state.get("caregiver_page", 1)
    page = max(1, min(page, total_pages))
    st.session_state.caregiver_page = page
    items = caregivers[(page - 1) * per_page : page * per_page]

    for cid, name, age, phone, days_str, times_str, special,\
        amin, amax, diligent, sociable, cheerful, warm, positive, observant in items:

        with sqlite3.connect("users.db", check_same_thread=False) as conn:
            img_row = conn.execute("""
                SELECT image_url FROM users
                WHERE id = (SELECT user_id FROM caregivers WHERE id=?)
            """, (cid,)).fetchone()
        img_url = img_row[0] if img_row else None

        st.markdown("---")
        col_img, col_txt = st.columns([1, 4])
        with col_img:
            if img_url:
                st.image(img_url, width=80)
            else:
                st.markdown("이미지 없음")

        with col_txt:
            st.markdown(f"**👩‍🍼 {name}** (만 {age}세)")
            st.write(f"• 연락처: `{phone}`")

            try:
                parsed = json.loads(times_str)
                kor_time = ", ".join(f"{t['start']}시–{t['end']}시"
                                     for t in parsed if isinstance(t, dict))
            except:
                kor_time = "미설정"

            st.markdown(
                f"""• <b>가능 요일:</b> {days_str or '미설정'}<br>
                    • <b>가능 시간대:</b> {kor_time}<br>
                    • <b>특수아동 케어:</b> {'가능' if special else '불가'}<br>
                    • <b>희망아동 연령:</b> {format_age(amin)} ~ {format_age(amax)}""",
                unsafe_allow_html=True
                )


            traits = {
                "꼼꼼함": diligent, "사교성": sociable, "쾌활함": cheerful,
                "따뜻함": warm, "긍정성": positive, "관찰력": observant
            }

            if all((v is None or v == 0) for v in traits.values()):
                with sqlite3.connect("users.db", check_same_thread=False) as conn:
                    vec = conn.execute("SELECT personality_traits_vector FROM caregivers WHERE id=?", (cid,)).fetchone()
                if vec and vec[0]:
                    vals = json.loads(vec[0])
                    keys = list(traits)
                    for i, k in enumerate(keys):
                        if i < len(vals): traits[k] = float(vals[i])

            df = pd.DataFrame(traits.items(), columns=["trait", "점수"])
            st.altair_chart(
                alt.Chart(df).mark_bar(color="#4caf50", size=14)
                   .encode(
                       x=alt.X("점수:Q", scale=alt.Scale(domain=[0, 1]), title="점수 (0~1)"),
                       y=alt.Y("trait:N", sort=list(traits), title="")
                   ),
                use_container_width=True
            )

            with sqlite3.connect("users.db", check_same_thread=False) as conn3:
                reviews = conn3.execute("""
                    SELECT parent_name, content, timestamp
                    FROM reviews
                    WHERE caregiver_id = ?
                    ORDER BY timestamp DESC
                """, (cid,)).fetchall()

            if reviews:
                st.markdown("**📝 후기**")
                for pname, content, ts in reviews:
                    st.write(f"- **{pname}님**: {content} _({ts[:19]})_")

    # 페이지 네비게이션
    

    col_prev, col_center, col_next = st.columns([2, 12, 2])

    with col_prev:
        if st.button("◀ 이전", key="prev_page_btn"):
            st.session_state.caregiver_page = max(1, st.session_state.caregiver_page - 1)
            st.rerun()

    with col_center:
        st.markdown(f"<p style='text-align:center;'></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;color:gray;'>페이지 {page} / {total_pages}</p>", unsafe_allow_html=True)

    with col_next:
        if st.button("다음 ▶", key="next_page_btn"):
            st.session_state.caregiver_page = min(total_pages, st.session_state.caregiver_page + 1)
            st.rerun()



    st.markdown(f"<p style='text-align:center;'></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'></p>", unsafe_allow_html=True)

    col_prev, col_center, col_next = st.columns([3, 2, 3])

    with col_prev:
        st.markdown(f"<p style='text-align:center;'></p>", unsafe_allow_html=True)

    with col_center:
        if st.button("🏠 홈으로 돌아가기"):
            st.session_state.page = "home"
            st.session_state.caregiver_page = 1
            st.rerun()

    with col_next:
        st.markdown(f"<p style='text-align:center;'></p>", unsafe_allow_html=True)






########################################
#
def page_caregiver_settings():
    # ────────────────────────────────────────────────
    # 1) 내비게이션 바 (뒤로 / 제목 / 홈)
    # ────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        if st.button("◀", key="back_to_home_from_settings"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        st.markdown("<h3 style='text-align: center;'>⚙️ 조건 설정</h3>", unsafe_allow_html=True)
    with col3:
        if st.button("🏠", key="home_from_settings"):
            st.session_state.page = "home"
            st.rerun()

    # ────────────────────────────────────────────────
    # 2) 페이지 전용 CSS (컨테이너 꽉 채우기 + 폼 간격)
    # ────────────────────────────────────────────────
    st.markdown("""
    <style>
      .block-container {
        min-height: 100vh !important;
        display: flex;
        flex-direction: column;
        padding-top: 2rem;
      }
      /* 폼 요소 간격 */
      .stForm > div {
        margin-bottom: 1.5rem !important;
      }
      /* 제출 버튼 가로 너비 */
      .stForm button[type="submit"] > button {
        width: 100% !important;
        padding: 0.75rem 0 !important;
        font-size: 1rem !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────
    # 3) 조건 입력 폼
    # ────────────────────────────────────────────────
    with st.form("caregiver_settings_form"):
        # (1) 돌봄 가능 연령 필터
        age_options = ["0~2세", "3~5세", "6세 이상", "전 연령"]
        st.multiselect(
            "돌봄 가능 연령 선택",
            options=age_options,
            default=st.session_state.get("filter_age", ["전 연령"]),
            key="filter_age"
        )

        # (2) 활동 가능 요일 필터
        day_options = ["월", "화", "수", "목", "금", "토", "일"]
        st.multiselect(
            "활동 가능 요일 선택",
            options=day_options,
            default=st.session_state.get("filter_days", day_options),
            key="filter_days"
        )

        # (3) 최대 1시간당 요금 필터 (단위는 레이블에 넣기)
        st.number_input(
            "최대 1시간당 요금 (원)",
            min_value=0,
            step=1000,
            value=st.session_state.get("filter_max_rate", 0),
            key="filter_max_rate",
            format="%d"  # 숫자만 포맷
        )

        # (4) 저장 버튼 — 반드시 폼 안에 있어야 합니다!
        submitted = st.form_submit_button("저장")

    # ────────────────────────────────────────────────
    # 4) 저장 후 처리
    # ────────────────────────────────────────────────
    if submitted:
        st.success("✅ 조건이 저장되었습니다.")
        # 저장된 조건을 다음 조회에 사용하도록 바로 돌보미 목록으로 이동
        st.session_state.page = "home"
        st.rerun()
#  
        
########################################


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

elif page == "caregivers":
    page_caregiver_list()
elif page=="caregiver_personality":
    page_caregiver_personality()
    
elif page == "caregiver_settings":
    page_caregiver_settings()

# 더 만들 페이지가 생기면 여기 아래에 elif 로 추가
