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
    "time", "user_name", "ip", "project", "purpose",
    "model", "quality", "resolution", "input_fidelity",
    "render_mode", "prompt_length", "cost_krw_est", "output_path",
]

# gpt-image-1 edit API 지원 해상도
SUPPORTED_SIZES = [
    "auto",
    "1536x1024",
    "2048x1152",
    "1024x1024",
    "1024x1536",
]

USD_TO_KRW = 1400

OPENAI_IMAGE_COST_USD = {
    "gpt-image-2": {
        "low": 0.005,
        "medium": 0.041,
        "high": 0.165,
        "auto": 0.041,
    },
    "gpt-image-1.5": {
        "low": 0.013,
        "medium": 0.05,
        "high": 0.2,
        "auto": 0.05,
    },
    "gpt-image-1": {
        "low": 0.016,
        "medium": 0.063,
        "high": 0.25,
        "auto": 0.063,
    },
    "gpt-image-1-mini": {
        "low": 0.006,
        "medium": 0.015,
        "high": 0.052,
        "auto": 0.015,
    },
}

# =========================
# 로그 유틸
# =========================

def normalize_log_df(df):
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[LOG_COLUMNS]

def read_log():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(LOG_PATH)
    df = normalize_log_df(df)
    df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    return df

def prepare_log_df(df):
    df = df.copy()
    df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")
    df["cost_krw_est"] = pd.to_numeric(df["cost_krw_est"], errors="coerce").fillna(0)
    df["prompt_length"] = pd.to_numeric(df["prompt_length"], errors="coerce").fillna(0)
    return df

def filter_log_by_period(df, period):
    df = prepare_log_df(df)
    now = datetime.now(timezone(timedelta(hours=9))).replace(tzinfo=None)
    if period == "오늘":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "최근 7일":
        start = now - timedelta(days=7)
    elif period == "이번 달":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        return df
    return df[df["time_dt"] >= start]

def write_log(row):
    new_row = {col: row.get(col, "") for col in LOG_COLUMNS}
    existing_df = read_log()
    new_df = pd.DataFrame([new_row], columns=LOG_COLUMNS)
    result_df = pd.concat([existing_df, new_df], ignore_index=True)
    normalize_log_df(result_df).to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

def delete_log_rows(indices_to_delete):
    if not os.path.exists(LOG_PATH):
        return
    df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    df = df.drop(indices_to_delete).reset_index(drop=True)
    df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

def get_client_ip():
    try:
        headers = st.context.headers
        ip = headers.get("x-forwarded-for", "")
        if ip:
            return ip.split(",")[0].strip()
        return headers.get("x-real-ip", "unknown").strip()
    except Exception:
        return "unknown"

def now_kst_str():
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")

def mask_ip(ip):
    if not ip or ip == "unknown":
        return "unknown"
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:12]

def estimate_cost_krw(model_name, quality, pass_count=1):
    usd = OPENAI_IMAGE_COST_USD.get(model_name, {}).get(quality, 0.04)
    return int(round(usd * USD_TO_KRW * pass_count))

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
# 이미지 전처리 (핵심 수정)
# =========================

def preprocess_image(pil_image):
    img = pil_image.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=6)
    buf.seek(0)
    buf.name = "input.png"

    max_bytes = 48 * 1024 * 1024

    if buf.getbuffer().nbytes > max_bytes:
        w, h = img.size
        scale = (max_bytes / buf.getbuffer().nbytes) ** 0.5 * 0.95
        new_w = max(16, int(w * scale))
        new_h = max(16, int(h * scale))

        img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True, compress_level=6)
        buf.seek(0)
        buf.name = "input.png"

    return buf

# =========================
# OpenAI 호출 (수정)
# =========================

def build_web_like_prompt(user_prompt):
    return f"""
You are editing the uploaded image, not generating a new unrelated image.

The uploaded image is the fixed base image and the only geometry source.

CRITICAL WEB-LIKE EDITING RULES:
- Preserve the exact canvas, crop, orientation, site boundary, road network, parcel boundaries, colored zone shapes, surrounding satellite context, and all spatial relationships.
- Do not invent a different city.
- Do not reinterpret the site layout.
- Do not replace the whole scene with a newly imagined aerial image.
- Only transform the colored zoning areas into photorealistic aerial architecture and landscape.
- Keep the camera strictly vertical top-down, 90-degree nadir.
- No oblique view.
- No tilted camera.
- No cinematic drone angle.
- No visible flat zoning colors in the final result.
- No painterly, blurry, CG, cartoon, or miniature-model look.
- The final result must look like a high-resolution real satellite / aerial masterplan visualization.

USER TASK:
{user_prompt}

FAILURE CONDITIONS:
- If the road network moves, disappears, bends, or changes, the result is wrong.
- If the site boundary changes, the result is wrong.
- If the output looks like a different neighborhood, the result is wrong.
- If the image becomes blurry or painterly, the result is wrong.
- If any flat zoning color remains visible, the result is wrong.
"""


def generate_image(client, model_name, prompt, input_image_pil, resolution, quality):
    input_buffer = preprocess_image(input_image_pil)
    full_prompt = build_web_like_prompt(prompt)

    edit_kwargs = {
        "model": model_name,
        "image": input_buffer,
        "prompt": full_prompt,
        "size": resolution,
        "quality": quality,
        "output_format": "png",
    }

    if model_name != "gpt-image-2":
        edit_kwargs["input_fidelity"] = "high"

    response = client.images.edit(**edit_kwargs)
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def build_refine_prompt():
    return """
Keep the image composition, road geometry, parcel boundaries, site boundary, terrain, and camera angle exactly the same.

Improve only visual realism:
- make the scene look like a real high-altitude aerial photograph
- improve sunlight, shadow consistency, roof materials, facade details, roads, trees, and ground textures
- remove artificial CG look, cartoon style, plastic textures, and repetitive patterns
- preserve all original spatial relationships and layout

Do not redesign the masterplan.
Do not move, rotate, resize, or reshape any road, block, parcel, or building mass.
"""


def build_polish_prompt():
    return """
Final polish pass.

Keep everything spatially identical.

Enhance only fine visual quality:
- more natural vegetation density
- more realistic roof details, solar panels, HVAC units, skylights, and rooftop gardens
- better material variation
- subtle atmospheric haze
- consistent photoreal aerial lighting
- remove remaining artificial edges, flat colors, or repeated patterns

The final result must look like a real professional aerial masterplan visualization.
No geometry changes.
"""


def run_render_pipeline(client, model_name, user_prompt, input_pil, resolution, quality):
    return generate_image(
        client=client,
        model_name=model_name,
        prompt=user_prompt,
        input_image_pil=input_pil,
        resolution=resolution,
        quality=quality,
    )


# =========================
# UI
# =========================

st.title("PlanVision AI - demo")
st.caption("Image API (images.edit) 기반 이미지 편집 파이프라인 | gpt-image-2는 input_fidelity 자동 제외")

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
        "이미지 모델 선택",
        [
            "gpt-image-2",
            "gpt-image-1.5",
            "gpt-image-1",
            "gpt-image-1-mini",
        ],
        index=0,
        help="ChatGPT 웹과 가장 가까운 결과를 노릴 때는 우선 최신 이미지 모델부터 테스트하세요. 모델별 지원 파라미터가 다를 수 있습니다."
    )

    st.divider()
    st.subheader("생성 옵션")

    resolution = st.selectbox(
        "Resolution",
        SUPPORTED_SIZES,
        index=0,
        help="auto = 입력 이미지 비율을 최대한 유지"
    )

    quality = st.selectbox(
        "Quality",
        ["high", "medium", "low", "auto"],
        index=0,
        help="최종본은 high 권장. gpt-image-2는 해상도와 품질에 따라 비용이 달라집니다."
    )

    input_fidelity = "high"

    render_mode = "원본 보존 생성 (1-pass)"
    pass_count = 1

    st.info("현재는 원본 보존을 위해 1-pass로 고정되어 있습니다.")

    st.divider()
    st.caption(f"예상 비용: {estimate_cost_krw(model_name, quality, pass_count):,}원 / 장")

    log_df_sidebar = read_log()
    st.download_button(
        "사용 로그 CSV 다운로드",
        data=log_df_sidebar.to_csv(index=False, encoding="utf-8-sig"),
        file_name="usage_log.csv",
        mime="text/csv"
    )

# =========================
# 메인 영역
# =========================

uploaded_image = st.file_uploader(
    "입력 이미지 업로드 (구역도 / 컬러맵)",
    type=["png", "jpg", "jpeg"]
)

prompt = st.text_area(
    "프롬프트",
    value="",
    placeholder="프롬프트를 입력하세요.",
    height=300
)

col1, col2 = st.columns(2)

input_pil = None

with col1:
    if uploaded_image:
        input_pil = Image.open(uploaded_image).convert("RGB")
        st.image(input_pil, caption=f"입력 이미지 ({input_pil.width}x{input_pil.height})", use_container_width=True)

with col2:
    run = st.button("조감도 생성", type="primary", use_container_width=True)

    output_path = ""

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

        error_message = ""

        try:
            client = get_client(api_key_input)

            with st.spinner("이미지 생성 중... (high quality 기준 최대 2분 정도 소요될 수 있음)"):
                result_image = run_render_pipeline(
                    client=client,
                    model_name=model_name,
                    user_prompt=prompt,
                    input_pil=input_pil,
                    resolution=resolution,
                    quality=quality,
                )

            st.image(result_image, caption="생성 결과", use_container_width=True)

            save_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_user = user_name.replace(" ", "_")
            safe_project = project_name.replace(" ", "_") if project_name else "no_project"
            output_filename = f"{save_time}_{safe_user}_{safe_project}_{model_name}_{resolution}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            result_image.save(output_path, format="PNG")

            output_buffer = BytesIO()
            result_image.save(output_buffer, format="PNG")

            st.success(f"저장 완료: {output_path}")
            st.download_button(
                "결과 이미지 다운로드",
                data=output_buffer.getvalue(),
                file_name=output_filename,
                mime="image/png",
                use_container_width=True
            )

        except Exception as e:
            error_message = str(e)
            st.error(f"오류 발생: {error_message}")

        finally:
            write_log({
                "time": now_kst_str(),
                "user_name": user_name,
                "ip": mask_ip(get_client_ip()),
                "project": project_name,
                "purpose": purpose,
                "model": model_name,
                "quality": quality,
                "resolution": resolution,
                "input_fidelity": input_fidelity,
                "prompt_length": len(prompt),
                "render_mode": render_mode,
                "cost_krw_est": estimate_cost_krw(model_name, quality, pass_count),
                "output_path": output_path,
            })

# =========================
# 로그 관리
# =========================

st.divider()
st.subheader("사용 로그 관리")

log_df = read_log()

if len(log_df) == 0:
    st.caption("아직 기록된 사용 로그가 없습니다.")
else:
    period = st.radio("조회 기간", ["오늘", "최근 7일", "이번 달", "전체"], horizontal=True)
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
        display_df = filtered_df.sort_values("time", ascending=False).reset_index()
        if len(display_df) == 0:
            st.caption("로그 없음")
        else:
            for i, row in display_df.iterrows():
                row_cols = st.columns([6, 1])
                with row_cols[0]:
                    st.write(
                        f"{row['time']} | {row['user_name']} | {row['project']} | "
                        f"{row['purpose']} | {row['model']} | {row['quality']} | "
                        f"{row['resolution']} | fidelity:{row['input_fidelity']} | "
                        f"{row['prompt_length']}자 | {row['cost_krw_est']}원"
                    )
                with row_cols[1]:
                    if st.button("❌", key=f"del_{row['index']}"):
                        delete_log_rows([row["index"]])
                        st.success("삭제 완료")
                        st.rerun()

    with tab2:
        user_summary = (
            filtered_df.groupby("user_name", dropna=False)
            .agg(호출수=("user_name", "count"), 추정비용=("cost_krw_est", "sum"), 평균프롬프트길이=("prompt_length", "mean"))
            .reset_index().rename(columns={"user_name": "사용자명"})
        )
        user_summary["추정비용"] = user_summary["추정비용"].astype(int)
        user_summary["평균프롬프트길이"] = user_summary["평균프롬프트길이"].fillna(0).astype(int)
        st.dataframe(user_summary.sort_values("호출수", ascending=False).reset_index(drop=True), use_container_width=True)

    with tab3:
        model_summary = (
            filtered_df.groupby("model", dropna=False)
            .agg(호출수=("model", "count"), 추정비용=("cost_krw_est", "sum"), 평균프롬프트길이=("prompt_length", "mean"))
            .reset_index().rename(columns={"model": "모델"})
        )
        model_summary["추정비용"] = model_summary["추정비용"].astype(int)
        model_summary["평균프롬프트길이"] = model_summary["평균프롬프트길이"].fillna(0).astype(int)
        st.dataframe(model_summary.sort_values("호출수", ascending=False).reset_index(drop=True), use_container_width=True)

    st.download_button(
        "현재 조회 로그 다운로드",
        data=filtered_df.drop(columns=["time_dt"], errors="ignore").to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"usage_log_{period}.csv",
        mime="text/csv"
    )

# =========================
# 갤러리
# =========================

st.divider()
st.subheader("생성 이미지 갤러리")

image_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")], reverse=True)

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
                    "다운로드", f,
                    file_name=img_file,
                    mime="image/png",
                    key=img_file
                )
