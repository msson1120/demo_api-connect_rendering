아래 수정만 하면 됩니다.

LOG_COLUMNS에 purpose 추가

기존:

LOG_COLUMNS = [
    "time",
    "user_name",
    "ip",
    "project",
    "model",
    "prompt_length",
    "cost_krw_est",
]

수정:

LOG_COLUMNS = [
    "time",
    "user_name",
    "ip",
    "project",
    "purpose",
    "model",
    "prompt_length",
    "cost_krw_est",
]
사이드바 사용자 정보에 활용목적 입력 추가

project_name = st.text_input(...) 바로 아래에 추가:

purpose = st.text_input("활용목적", placeholder="예: QBS")
프롬프트 기본값 제거

기존 default_prompt 전체 삭제하고, prompt = st.text_area(...)를 아래로 교체:

prompt = st.text_area(
    "프롬프트",
    value="",
    placeholder="AI Studio처럼 사용할 프롬프트를 입력하세요.",
    height=300
)
생성 버튼 클릭 시 활용목적/프롬프트 검증 추가

if not user_name: 아래 근처에 추가:

if not purpose:
    st.warning("활용목적을 입력하세요.")
    st.stop()

if not prompt.strip():
    st.warning("프롬프트를 입력하세요.")
    st.stop()
로그 저장 부분에 purpose 추가

기존 write_log({...}) 안에 추가:

"purpose": purpose,

최종 형태는 이런 식입니다.

write_log({
    "time": now_kst_str(),
    "user_name": user_name,
    "ip": masked_ip,
    "project": project_name,
    "purpose": purpose,
    "model": model_name,
    "prompt_length": len(prompt),
    "cost_krw_est": estimate_cost_krw(model_name),
})
상세 로그 표시 컬럼에도 purpose 추가

기존:

show_cols = [
    "time",
    "user_name",
    "ip",
    "project",
    "model",
    "prompt_length",
    "cost_krw_est",
]

수정:

show_cols = [
    "time",
    "user_name",
    "ip",
    "project",
    "purpose",
    "model",
    "prompt_length",
    "cost_krw_est",
]
