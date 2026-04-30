import streamlit as st
import pandas as pd
import time
import os
import hashlib
from datetime import datetime, timezone, timedelta
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
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_COLUMNS = [
    "time",
    "user_name",
    "ip",
    "project",
    "purpose",
    "model",
    "prompt_length",
    "cost_krw_est",
    "output_path",
]


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


def prepare_log_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "time" in df.columns:
        df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["time_dt"] = pd.NaT

    df["cost_krw_est"] = pd.to_numeric(df["cost_krw_est"], errors="coerce").fillna(0)
    df["prompt_length"] = pd.to_numeric(df["prompt_length"], errors="coerce").fillna(0)

    return df


def filter_log_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    df = prepare_log_df(df)

    now = datetime.now(timezone(timedelta(hours=9))).replace(tzinfo=None)

    if period == "오늘":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "최근 7일":
        start = now - timedelta(days=7)
    elif period == "이번 달":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "전체":
        return df
    else:
        return df

    return df[df["time_dt"] >= start]


def write_log(row: dict):
    new_row = {}

    for col in LOG_COLUMNS:
        new_row[col] = row.get(col, "")

    existing_df = read_log()
    new_df = pd.DataFrame([new_row], columns=LOG_COLUMNS)

    result_df = pd.concat([existing_df, new_df], ignore_index=True)
    result_df = normalize_log_df(result_df)

    result_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


def delete_log_rows(indices_to_delete):
    if not os.path.exists(LOG_PATH):
        return

    df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    df = df.drop(indices_to_delete)
    df = df.reset_index(drop=True)
    df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


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


def now_kst_str():
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")


def mask_ip(ip: str):
    if not ip or ip == "unknown":
        return "unknown"

    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:12]


USD_TO_KRW = 1400

IMAGE_OUTPUT_COST_USD = {
    "gemini-2.5-flash-image": {
        "1K": 0.039,
        "2K": 0.039,
        "4K": 0.039,
    },
    "gemini-3.1-flash-image-preview": {
        "1K": 0.067,
        "2K": 0.067,
        "4K": 0.151,
    },
    "gemini-3-pro-image-preview": {
        "1K": 0.134,
        "2K": 0.134,
        "4K": 0.240,
    },
}

TEXT_INPUT_COST_PER_1M_USD = {
    "gemini-2.5-flash-image": 0.10,
    "gemini-3.1-flash-image-preview": 0.25,
    "gemini-3-pro-image-preview": 2.00,
}


def estimate_cost_krw(model_name: str, prompt_length: int, resolution: str):
    input_tokens_est = max(int(prompt_length / 3.5), 1)

    text_input_usd = (
        input_tokens_est / 1_000_000
    ) * TEXT_INPUT_COST_PER_1M_USD.get(model_name, 0.25)

    image_output_usd = IMAGE_OUTPUT_COST_USD.get(
        model_name, {}
    ).get(resolution, 0.067)

    total_usd = text_input_usd + image_output_usd
    return int(round(total_usd * USD_TO_KRW))


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

def generate_image(client, model_name, prompt, input_image, resolution, top_p):
    config_kwargs = {
        "response_modalities": ["TEXT", "IMAGE"],
        "top_p": top_p,
    }

    image_config_kwargs = {}

    if resolution == "1K":
        image_config_kwargs["image_size"] = "1K"
    elif resolution == "2K":
        image_config_kwargs["image_size"] = "2K"
    elif resolution == "4K":
        image_config_kwargs["image_size"] = "4K"

    if image_config_kwargs:
        config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

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
    purpose = st.text_input("활용목적", placeholder="예: QBS")

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
        ["1K", "2K", "4K"],
        index=0,
        help="모델에 따라 4K는 지원되지 않을 수 있습니다."
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

prompt = st.text_area(
    "프롬프트",
    value="",
    placeholder="AI Studio처럼 사용할 프롬프트를 입력하세요.",
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

        if not purpose:
            st.warning("활용목적을 입력하세요.")
            st.stop()

        if not prompt.strip():
            st.warning("프롬프트를 입력하세요.")
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
                    top_p=top_p
                )

            if result_image:
                st.image(result_image, caption="생성 결과", use_container_width=True)

                save_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_user = user_name.replace(" ", "_")
                safe_project = project_name.replace(" ", "_") if project_name else "no_project"

                output_filename = f"{save_time}_{safe_user}_{safe_project}_{model_name}_{resolution}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                result_image.save(output_path, format="PNG")

                output_buffer = BytesIO()
                result_image.save(output_buffer, format="PNG")

                st.success(f"이미지가 자동 저장되었습니다: {output_path}")

                st.download_button(
                    "결과 이미지 다운로드",
                    data=output_buffer.getvalue(),
                    file_name=output_filename,
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
                "time": now_kst_str(),
                "user_name": user_name,
                "ip": masked_ip,
                "project": project_name,
                "purpose": purpose,
                "model": model_name,
                "prompt_length": len(prompt),
                "cost_krw_est": estimate_cost_krw(model_name, len(prompt), resolution),
                "output_path": output_path if "output_path" in locals() else "",
            })


st.divider()
st.subheader("사용 로그 관리")

log_df = read_log()

if len(log_df) == 0:
    st.caption("아직 기록된 사용 로그가 없습니다.")
else:
    period = st.radio(
        "조회 기간",
        ["오늘", "최근 7일", "이번 달", "전체"],
        horizontal=True
    )

    filtered_df = filter_log_by_period(log_df, period)

    total_calls = len(filtered_df)
    unique_users = filtered_df["user_name"].nunique()
    total_cost = int(filtered_df["cost_krw_est"].sum())
    avg_prompt_length = int(filtered_df["prompt_length"].mean()) if total_calls > 0 else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("호출 수", f"{total_calls:,}회")
    c2.metric("사용자 수", f"{unique_users:,}명")
    c3.metric("추정 비용", f"{total_cost:,}원")
    c4.metric("평균 프롬프트 길이", f"{avg_prompt_length:,}자")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["상세 로그", "사용자별 요약", "모델별 요약"])

    with tab1:
        log_df_display = log_df.reset_index().rename(columns={"index": "row_id"})

        selected_rows = st.multiselect(
            "삭제할 로그 선택",
            options=log_df_display["row_id"],
            format_func=lambda x: f"{log_df_display.loc[x, 'time']} | {log_df_display.loc[x, 'user_name']} | {log_df_display.loc[x, 'project']}"
        )

        del_col, _ = st.columns([1, 5])

        with del_col:
            if st.button("선택 삭제"):
                if selected_rows:
                    delete_log_rows(selected_rows)
                    st.success("삭제 완료")
                    st.rerun()
                else:
                    st.warning("삭제할 행을 선택하세요.")

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

        st.dataframe(
            filtered_df[show_cols].sort_values("time", ascending=False).reset_index(drop=True),
            use_container_width=True
        )

    with tab2:
        user_summary = (
            filtered_df
            .groupby("user_name", dropna=False)
            .agg(
                호출수=("user_name", "count"),
                추정비용=("cost_krw_est", "sum"),
                평균프롬프트길이=("prompt_length", "mean"),
            )
            .reset_index()
            .rename(columns={"user_name": "사용자명"})
        )

        user_summary["추정비용"] = user_summary["추정비용"].astype(int)
        user_summary["평균프롬프트길이"] = user_summary["평균프롬프트길이"].fillna(0).astype(int)

        st.dataframe(
            user_summary.sort_values("호출수", ascending=False).reset_index(drop=True),
            use_container_width=True
        )

    with tab3:
        model_summary = (
            filtered_df
            .groupby("model", dropna=False)
            .agg(
                호출수=("model", "count"),
                추정비용=("cost_krw_est", "sum"),
                평균프롬프트길이=("prompt_length", "mean"),
            )
            .reset_index()
            .rename(columns={"model": "모델"})
        )

        model_summary["추정비용"] = model_summary["추정비용"].astype(int)
        model_summary["평균프롬프트길이"] = model_summary["평균프롬프트길이"].fillna(0).astype(int)

        st.dataframe(
            model_summary.sort_values("호출수", ascending=False).reset_index(drop=True),
            use_container_width=True
        )

    st.download_button(
        "현재 조회 로그 다운로드",
        data=filtered_df.drop(columns=["time_dt"], errors="ignore").to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"usage_log_{period}.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("생성 이미지 갤러리")

image_files = sorted(
    [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")],
    reverse=True
)

if len(image_files) == 0:
    st.caption("아직 생성된 이미지가 없습니다.")
else:
    cols = st.columns(3)

    for idx, img_file in enumerate(image_files[:30]):
        img_path = os.path.join(OUTPUT_DIR, img_file)

        with cols[idx % 3]:
            st.image(img_path, use_container_width=True)
            st.caption(img_file)

            with open(img_path, "rb") as f:
                st.download_button(
                    "다운로드",
                    f,
                    file_name=img_file,
                    mime="image/png",
                    key=img_file
                )
