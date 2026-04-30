import streamlit as st
import pandas as pd
import time
import os
import hashlib
from datetime import datetime
from PIL import Image
from io import BytesIO

from google import genai
from google.genai import types


# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="PlanVision AI - demo",
    page_icon="🏙️",
    layout="wide"
)

LOG_PATH = "usage_log.csv"

LOG_COLUMNS = [
    "time",
    "user_name",
    "ip",
    "project",
    "model",
    "prompt_length",
    "cost_krw_est",
]

COST_PER_CALL_KRW = {
    "gemini-3.1-flash-image-preview": 50,
    "gemini-3-pro-image-preview": 150,
    "gemini-2.5-flash-image": 40,
}


# =========================
# 로그 유틸
# =========================

def normalize_log_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[LOG_COLUMNS]


def read_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)

    try:
        df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(LOG_PATH)

    df = normalize_log_df(df)

    # 기존 CSV 컬럼 구조가 다르면 즉시 새 구조로 재저장
    df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    return df


def write_log(row: dict):
    new_row = {}

    for col in LOG_COLUMNS:
        new_row[col] = row.get(col, "")

    existing_df = read_log()
    new_df = pd.DataFrame([new_row], columns=LOG_COLUMNS)

    result_df = pd.concat([existing_df, new_df], ignore_index=True)
    result_df = normalize_log_df(result_df)

    result_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


def get_client_ip():
    try:
        headers = st.context.headers

        ip = headers.get("x-forwarded-for", "")
        if ip:
            return ip.split(",")[0].strip()

        ip = headers.get("x-real-ip", "")
        if ip:
            return ip.strip()

        return "unknown"
    except Exception:
        return "unknown"


def mask_ip(ip: str):
    if not ip or ip == "unknown":
        return "unknown"

    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:12]


def estimate_cost_krw(model_name: str):
    return COST_PER_CALL_KRW.get(model_name, 0)


# =========================
# Gemini Client
# =========================

def get_client(api_key_input):
    api_key = api_key_input or st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI API KEY를 입력하세요.")
        st.stop()

    return genai.Client(api_key=api_key)


# =========================
# 이미지 처리
# =========================

def uploaded_file_to_pil(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


def extract_image_from_response(response):
    if not response.candidates:
        return None

    parts = response.candidates[0].content.parts

    for part in parts:
        if getattr(part, "inline_data", None):
            image_bytes = part.inline_data.data
            return Image.open(BytesIO(image_bytes))

    return None


# =========================
# Gemini 호출
# =========================

def generate_image(client, model_name, prompt, input_image, resolution, thinking_level, top_p):
    config_kwargs = {
        "response_modalities": ["TEXT", "IMAGE"],
        "top_p": top_p,
    }

    image_config_kwargs = {}

    if resolution == "1K":
        image_config_kwargs["image_size"] = "1K"
    elif resolution == "2K":
        image_config_kwargs["image_size"] = "2K"

    if image_config_kwargs:
        config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

    if thinking_level != "none":
        thinking_budget_map = {
            "low": 1024,
            "medium": 4096,
            "high": 8192,
        }

        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget_map[thinking_level]
        )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, input_image],
            config=types.GenerateContentConfig(**config_kwargs)
        )

    except Exception:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, input_image],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

    return extract_image_from_response(response)


# =========================
# UI
# =========================

st.title("PlanVision AI - demo")
st.caption("AI Studio처럼 사용하되, 사용자별 사용 여부와 호출 로그를 남기는 데모 프로그램입니다.")

with st.sidebar:
    st.subheader("API 설정")

    api_key_input = st.text_input(
        "GEMINI API KEY",
        type="password",
        placeholder="AI Studio에서 발급받은 키 입력"
    )

    st.header("사용자 정보")

    user_name = st.text_input("사용자명", placeholder="예: 홍길동")
    project_name = st.text_input("프로젝트명", placeholder="예: 의왕도첨산단")

    model_name = st.selectbox(
        "모델 선택",
        [
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
            "gemini-2.5-flash-image"
        ],
        index=0
    )

    st.divider()
    st.subheader("생성 옵션")

    resolution = st.selectbox(
        "Resolution",
        ["1K", "2K"],
        index=0
    )

    thinking_level = st.selectbox(
        "Thinking level",
        ["none", "low", "medium", "high"],
        index=0
    )

    top_p = st.slider(
        "Top P",
        min_value=0.1,
        max_value=1.0,
        value=0.95,
        step=0.05
    )

    st.divider()
    st.subheader("로그")

    log_df_sidebar = read_log()

    st.download_button(
        "사용 로그 CSV 다운로드",
        data=log_df_sidebar.to_csv(index=False, encoding="utf-8-sig"),
        file_name="usage_log.csv",
        mime="text/csv"
    )


uploaded_image = st.file_uploader(
    "입력 이미지 업로드",
    type=["png", "jpg", "jpeg"]
)

default_prompt = """
Transform the input image into a photorealistic 3D oblique aerial architectural rendering.

STRICT RULES:
- Use the input image as the only geometry source.
- Preserve roads, parcels, terrain, water bodies, and site boundary.
- Do not change the outside boundary.
- Remove all zoning colors.
- Convert colored zones into realistic architecture or landscape.
- Keep the camera angle, perspective, and layout consistent.
- Output must look like a realistic urban masterplan aerial render.
"""

prompt = st.text_area(
    "프롬프트",
    value=default_prompt.strip(),
    height=300
)

col1, col2 = st.columns(2)

input_pil = None

with col1:
    if uploaded_image:
        input_pil = uploaded_file_to_pil(uploaded_image)
        st.image(input_pil, caption="입력 이미지", use_container_width=True)

with col2:
    run = st.button("조감도 생성", type="primary", use_container_width=True)

    if run:
        if not user_name:
            st.warning("사용자명을 입력하세요.")
            st.stop()

        if not uploaded_image:
            st.warning("입력 이미지를 업로드하세요.")
            st.stop()

        start_time = time.time()
        error_message = ""

        try:
            client = get_client(api_key_input)

            with st.spinner("이미지 생성 중..."):
                result_image = generate_image(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    input_image=input_pil,
                    resolution=resolution,
                    thinking_level=thinking_level,
                    top_p=top_p
                )

            if result_image:
                st.image(result_image, caption="생성 결과", use_container_width=True)

                output_buffer = BytesIO()
                result_image.save(output_buffer, format="PNG")

                st.download_button(
                    "결과 이미지 다운로드",
                    data=output_buffer.getvalue(),
                    file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                error_message = "응답에서 이미지를 찾지 못했습니다."
                st.error(error_message)

        except Exception as e:
            error_message = str(e)
            st.error(error_message)

        finally:
            raw_ip = get_client_ip()
            masked_ip = mask_ip(raw_ip)

            write_log({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_name": user_name,
                "ip": masked_ip,
                "project": project_name,
                "model": model_name,
                "prompt_length": len(prompt),
                "cost_krw_est": estimate_cost_krw(model_name),
            })

            st.rerun()


st.divider()
st.subheader("최근 사용 로그")

log_df = read_log()

if len(log_df) > 0:
    st.dataframe(
        log_df.tail(20).reset_index(drop=True),
        use_container_width=True
    )
else:
    st.caption("아직 기록된 사용 로그가 없습니다.")
