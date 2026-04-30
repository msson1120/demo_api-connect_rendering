import streamlit as st
import pandas as pd
import time
import uuid
import os
from datetime import datetime
from PIL import Image
from io import BytesIO

from google import genai
from google.genai import types


# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="AI 조감도 생성 데모",
    page_icon="🏙️",
    layout="wide"
)

LOG_PATH = "usage_log.csv"


# =========================
# Gemini Client
# =========================

def get_client():
    api_key = api_key_input or st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI API KEY를 입력하세요.")
        st.stop()

    return genai.Client(api_key=api_key)


# =========================
# 로그 저장
# =========================

def write_log(row: dict):
    df = pd.DataFrame([row])

    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


# =========================
# 이미지 변환
# =========================

def uploaded_file_to_pil(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


def extract_image_from_response(response):
    for part in response.candidates[0].content.parts:
        if getattr(part, "inline_data", None):
            image_bytes = part.inline_data.data
            return Image.open(BytesIO(image_bytes))
    return None


# =========================
# Gemini 호출
# =========================

def generate_image(client, model_name, prompt, input_image):
    response = client.models.generate_content(
        model=model_name,
        contents=[
            prompt,
            input_image
        ],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        )
    )

    return extract_image_from_response(response)


# =========================
# UI
# =========================

st.title("구역계 기반 AI 조감도 생성 데모")
st.caption("AI Studio처럼 사용하되, 사용자별 사용 여부와 호출 로그를 남기는 Streamlit 데모 엔진입니다.")

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

    st.subheader("로그")
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        st.download_button(
            "사용 로그 CSV 다운로드",
            data=log_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="usage_log.csv",
            mime="text/csv"
        )
    else:
        st.info("아직 로그가 없습니다.")


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

        request_id = str(uuid.uuid4())
        start_time = time.time()
        success = False
        error_message = ""

        try:
            client = get_client()

            with st.spinner("이미지 생성 중..."):
                result_image = generate_image(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    input_image=input_pil
                )

            elapsed_sec = round(time.time() - start_time, 2)

            if result_image:
                success = True
                st.image(result_image, caption="생성 결과", use_container_width=True)

                output_buffer = BytesIO()
                result_image.save(output_buffer, format="PNG")

                st.download_button(
                    "결과 이미지 다운로드",
                    data=output_buffer.getvalue(),
                    file_name=f"result_{request_id}.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                error_message = "응답에서 이미지를 찾지 못했습니다."
                st.error(error_message)

        except Exception as e:
            elapsed_sec = round(time.time() - start_time, 2)
            error_message = str(e)
            st.error(error_message)

        finally:
            write_log({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": request_id,
                "user_name": user_name,
                "project_name": project_name,
                "model": model_name,
                "input_filename": uploaded_image.name if uploaded_image else "",
                "prompt_length": len(prompt),
                "success": success,
                "elapsed_sec": elapsed_sec,
                "error_message": error_message
            })


st.divider()

st.subheader("최근 사용 로그")

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    st.dataframe(log_df.tail(20), use_container_width=True)
else:
    st.caption("아직 기록된 사용 로그가 없습니다.")
