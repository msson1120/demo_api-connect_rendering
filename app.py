아래는 현재 Gemini 코드에서 OpenAI GPT 이미지 API로 갈아타는 수정부분입니다.

requirements.txt 수정

기존:

google-genai

삭제하고 아래 추가:

openai

최종 예:

streamlit
pandas
pillow
openai
import 교체

기존 삭제:

from google import genai
from google.genai import types

아래 추가:

import base64
from openai import OpenAI
Gemini Client 함수 교체

기존 get_client() 함수 전체 삭제 후 교체:

def get_client(api_key_input):
    api_key = api_key_input or st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI API KEY를 입력하세요.")
        st.stop()

    return OpenAI(api_key=api_key)
사이드바 API 입력창 문구 수정

기존:

api_key_input = st.text_input(
    "GEMINI API KEY",
    type="password",
    placeholder="AI Studio에서 발급받은 키 입력"
)

수정:

api_key_input = st.text_input(
    "OPENAI API KEY",
    type="password",
    placeholder="OpenAI Platform에서 발급받은 API 키 입력"
)
모델 선택 교체

기존:

model_name = st.selectbox(
    "모델 선택",
    [
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image"
    ],
    index=0
)

수정:

model_name = st.selectbox(
    "모델 선택",
    [
        "gpt-image-1",
    ],
    index=0
)
생성 옵션 교체

기존 Resolution, Top P 부분을 아래로 교체하세요.

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

top_p, top_k, thinking_level은 삭제하세요.
이미지 생성/편집 API에서는 Top K 유지 불가입니다.

Gemini용 이미지 추출 함수 삭제

아래 함수는 삭제해도 됩니다.

def extract_image_from_response(response):
    ...
generate_image() 함수 전체 교체

기존 generate_image() 전체 삭제 후 아래로 교체:

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
generate_image 호출부 교체

기존:

result_image = generate_image(
    client=client,
    model_name=model_name,
    prompt=prompt,
    input_image=input_pil,
    resolution=resolution,
    top_p=top_p
)

수정:

result_image = generate_image(
    client=client,
    model_name=model_name,
    prompt=prompt,
    input_image=input_pil,
    resolution=resolution,
    quality=quality,
    input_fidelity=input_fidelity
)
자동저장 파일명에서 resolution은 그대로 사용 가능

기존:

output_filename = f"{save_time}_{safe_user}_{safe_project}_{model_name}_{resolution}.png"

그대로 둬도 됩니다.

비용 추정 함수 교체

기존 estimate_cost_krw() 삭제 후 아래로 교체:

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

그리고 로그 저장 부분 기존:

"cost_krw_est": estimate_cost_krw(model_name, len(prompt), resolution),

수정:

"cost_krw_est": estimate_cost_krw(model_name, quality),

결론:

Top K: GPT 이미지 API에서 불가
Top P: 이미지 편집 API에서는 일단 빼는 게 맞음
대신 조정 가능:
- size
- quality
- input_fidelity

지금 목적이 “조감도 결과 품질”이면 input_fidelity="high"와 quality="high" 조합부터 테스트하세요.
