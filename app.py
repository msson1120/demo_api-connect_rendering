import streamlit as st
import pandas as pd
import time
import os
import hashlib
from datetime import datetime, timezone, timedelta
from PIL import Image
from io import BytesIO

import base64
from openai import OpenAI


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

OPENAI_IMAGE_COST_USD = {
    "gpt-image-1": {
        "low": 0.01,
        "medium": 0.04,
        "high": 0.17,
        "auto": 0.04,
    }
}


def estimate_cost_krw(model_name: str, quality: str):
    usd = OPENAI_IMAGE_COST_USD.get(model_name, {}).get(quality, 0.04)
    return int(round(usd * USD_TO_KRW))


# =========================
# OpenAI Client
# =========================

def get_client(api_key_input):
    api_key = api_key_input or st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI API KEY를 입력하세요.")
        st.stop()

    return OpenAI(api_key=api_key)


# =========================
# 이미지 처리
# =========================

def uploaded_file_to_pil(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


# =========================
# OpenAI 호출
# =========================

def generate_image(client, model_name, prompt, input_image, resolution, quality, input_fidelity):
    input_buffer = BytesIO()
    input_image.save(input_buffer, format="PNG")
    input_buffer.seek(0)
    input_buffer.name = "input.png"

    response = client.images.edit(
        model=model_name,
        image=input_buffer,
        prompt=prompt,
        size=resolution,
        quality=quality,
        input_fidelity=input_fidelity,
        n=1,
    )

    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    return Image.open(BytesIO(image_bytes)).convert("RGB")


# =========================
# UI
# =========================

st.title("PlanVision AI - demo")
st.caption("AI Studio처럼 사용하되, 사용자별 사용 여부와 호출 로그를 남기는 데모 프로그램입니다.")

with st.sidebar:
    st.subheader("API 설정")

    api_key_input = st.text_input(
        "OPENAI API KEY",
        type="password",
        placeholder="OpenAI Platform에서 발급받은 API 키 입력"
    )

    st.header("사용자 정보")

    user_name = st.text_input("사용자명", placeholder="예: 홍길동")
    project_name = st.text_input("프로젝트명", placeholder="예: 의왕도첨산단")
    purpose = st.text_input("활용목적", placeholder="예: QBS")

    model_name = st.selectbox(
        "모델 선택",
        ["gpt-image-1"],
        index=0
    )

    st.divider()
    st.subheader("생성 옵션")

    resolution = st.selectbox(
        "Resolution",
        ["1024x1024", "1536x1024", "1024x1536"],
        index=1,
        help="GPT 이미지 API는 지정된 size 옵션을 사용합니다."
    )

    quality = st.selectbox(
        "Quality",
        ["auto", "low", "medium", "high"],
        index=0
    )

    input_fidelity = st.selectbox(
        "Input Fidelity",
        ["low", "high"],
        index=1,
        help="입력 이미지 유지 강도를 조정합니다. 구역계/배치도 기반이면 high 권장."
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
                    quality=quality,
                    input_fidelity=input_fidelity
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
                "cost_krw_est": estimate_cost_krw(model_name, quality),
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
        st.subheader("상세 로그 (행별 삭제)")

        display_df = filtered_df.sort_values("time", ascending=False).reset_index()

        if len(display_df) == 0:
            st.caption("로그 없음")
        else:
            for i, row in display_df.iterrows():
                row_cols = st.columns([6, 1])

                with row_cols[0]:
                    st.write(
                        f"{row['time']} | {row['user_name']} | {row['project']} | {row['purpose']} | {row['model']} | {row['prompt_length']}자 | {row['cost_krw_est']}원"
                    )

                with row_cols[1]:
                    if st.button("❌", key=f"del_{row['index']}"):
                        delete_log_rows([row["index"]])
                        st.success("삭제 완료")
                        st.rerun()

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
