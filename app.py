"""H형강 정지 이미지 자동 카운팅 데모 (Ultralytics YOLO OBB + Streamlit)."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = APP_DIR / "best.pt"
BLUE = (37, 99, 235)


st.set_page_config(
    page_title="H형강 자동 카운팅 | Steel Vision",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {background: #f5f7fb; color: #15253f;}
    .block-container {max-width: 1180px; padding-top: 2.1rem; padding-bottom: 3rem;}
    [data-testid="stHeader"] {background: transparent;}
    .hero {
        background: linear-gradient(120deg, #102745 0%, #193e71 65%, #2757a0 100%);
        border-radius: 19px;
        padding: 2.1rem 2.35rem 2.25rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 11px 32px rgba(12, 35, 70, .13);
    }
    .hero-label {color: #9bcaff; font-size: .84rem; letter-spacing: .12em; font-weight: 800;}
    .hero h1 {color: white; font-size: clamp(1.7rem, 3vw, 2.6rem); margin: .6rem 0 .7rem; font-weight: 800;}
    .hero p {color: #dceafd; margin: 0; line-height: 1.8;}
    .section-label {font-size: 1.16rem; font-weight: 800; color: #173156; margin: .1rem 0 1rem;}
    .result-box {
        background: #e9f1ff; border: 1px solid #c9dcff; border-radius: 14px;
        padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    }
    .result-label {font-size: .92rem; color: #33547e; font-weight: 650;}
    .result-value {font-size: 3.35rem; color: #1757c6; font-weight: 850; letter-spacing: -.04em; line-height: 1.45;}
    .hint {font-size: .85rem; color: #64748b; line-height: 1.75;}
    div.stButton > button[kind="primary"] {
        background: #215cc5; border-color: #215cc5; border-radius: 10px;
        font-weight: 750; min-height: 3rem;
    }
    [data-testid="stFileUploader"] section {border-radius: 12px;}
    [data-testid="stImage"] img {border-radius: 12px;}
    @media (max-width: 640px) {.hero {padding: 1.4rem 1.2rem;} .block-container {padding-top: 1rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="학습된 AI 모델을 불러오는 중입니다...")
def load_model(weights_path: str) -> YOLO:
    """Download-free, process-wide caching of the user-trained checkpoint."""
    return YOLO(weights_path)


def draw_obb(image: Image.Image, polygons: np.ndarray) -> Image.Image:
    """Draw easily distinguishable blue rotated bounding boxes and object indexes."""
    annotated = image.copy()
    painter = ImageDraw.Draw(annotated)
    line_width = max(3, round(min(image.size) / 190))

    for number, polygon in enumerate(polygons, start=1):
        points = [tuple(map(float, point)) for point in polygon]
        painter.line(points + [points[0]], fill=BLUE, width=line_width, joint="curve")

        # Start the index tag from the leftmost rotated-box corner.
        x = max(0, min(int(min(point[0] for point in points)), image.width - 1))
        y = max(0, min(int(min(point[1] for point in points)), image.height - 1))
        label = f"#{number}"
        tag_width = max(38, 13 + 9 * len(label))
        tag_height = 21
        y0 = min(max(0, y - tag_height), max(0, image.height - tag_height))
        x0 = min(x, max(0, image.width - tag_width))
        painter.rounded_rectangle(
            [x0, y0, x0 + tag_width, y0 + tag_height],
            radius=5,
            fill=BLUE,
        )
        painter.text((x0 + 8, y0 + 3), label, fill="white")

    return annotated


def analyze_image(image: Image.Image, threshold: float, iou: float) -> dict:
    model = load_model(str(WEIGHTS_PATH))
    prediction = model.predict(
        source=np.asarray(image),
        imgsz=640,
        conf=threshold,
        iou=iou,
        max_det=300,
        verbose=False,
    )[0]
    if prediction.obb is None:
        raise ValueError("업로드한 모델이 OBB(회전 경계 상자) 모델인지 확인해 주세요.")

    polygons = prediction.obb.xyxyxyxy.cpu().numpy()
    confidences = prediction.obb.conf.cpu().numpy()

    return {
        "count": int(len(polygons)),
        "image": draw_obb(image, polygons),
        "mean_confidence": float(confidences.mean()) if len(confidences) else None,
    }


st.markdown(
    """
    <section class="hero">
        <div class="hero-label">STEEL VISION · AI INSPECTION DEMO</div>
        <h1>H형강 자동 카운팅</h1>
        <p>생산라인 사진 한 장을 올리면 학습된 YOLO OBB 모델이<br>
        H형강을 찾아 파란색으로 표시하고 개수를 계산합니다.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not WEIGHTS_PATH.is_file():
    st.error("모델 파일(best.pt)이 없습니다. app.py와 같은 폴더에 best.pt를 넣어 주세요.")
    st.stop()

left_col, right_col = st.columns([1, 1.12], gap="large")

with left_col:
    st.markdown('<div class="section-label">① 이미지 업로드</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "H형강 생산라인 사진 선택",
        type=["jpg", "jpeg", "png", "webp"],
        help="학습 데이터와 비슷한 생산라인 상부 촬영 사진을 권장합니다.",
    )
    source_image = None
    image_hash = None

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        image_hash = hashlib.sha256(raw_bytes).hexdigest()
        try:
            source_image = ImageOps.exif_transpose(
                Image.open(BytesIO(raw_bytes))
            ).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"사진을 열 수 없습니다. 다른 이미지로 시도해 주세요. ({exc})")

        if source_image is not None:
            st.image(source_image, caption="업로드한 원본 사진", use_container_width=True)

    threshold = st.slider(
        "탐지 신뢰도 기준",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="높이면 잘못 탐지한 형강을 줄일 수 있지만 일부 형강을 놓칠 수 있습니다.",
    )
    with st.expander("고급 설정"):
        iou_threshold = st.slider(
            "중복 탐지 제거 기준(IoU)",
            min_value=0.10,
            max_value=0.90,
            value=0.70,
            step=0.05,
        )

    requested = st.button(
        "AI로 형강 개수 분석하기",
        type="primary",
        use_container_width=True,
        disabled=source_image is None,
    )
    st.markdown(
        '<p class="hint">※ 정지 이미지 한 장에 보이는 형강 수를 계산합니다. '
        '움직이는 형강의 누적 생산량을 추적하는 기능은 아닙니다.</p>',
        unsafe_allow_html=True,
    )

current_key = (image_hash, threshold, iou_threshold)
if requested and source_image is not None:
    try:
        with st.spinner("AI가 사진에서 H형강을 찾는 중입니다..."):
            output = analyze_image(source_image, threshold, iou_threshold)
        st.session_state["last_analysis"] = {"key": current_key, **output}
    except Exception as exc:
        st.session_state.pop("last_analysis", None)
        st.error(f"모델 실행 중 오류가 발생했습니다: {exc}")

with right_col:
    st.markdown('<div class="section-label">② AI 분석 결과</div>', unsafe_allow_html=True)
    saved_output = st.session_state.get("last_analysis")

    if saved_output is not None and saved_output["key"] == current_key:
        count = saved_output["count"]
        st.markdown(
            f'<div class="result-box"><div class="result-label">탐지된 H형강</div>'
            f'<div class="result-value">{count}개</div></div>',
            unsafe_allow_html=True,
        )
        st.image(
            saved_output["image"],
            caption="파란색 경계선: AI가 인식한 형강",
            use_container_width=True,
        )

        confidence = saved_output["mean_confidence"]
        if confidence is not None:
            st.caption(f"탐지 객체의 평균 신뢰도: {confidence:.1%} (정확도와는 다른 지표)")

        buffer = BytesIO()
        saved_output["image"].save(buffer, format="PNG")
        st.download_button(
            "📥 분석 결과 이미지 다운로드",
            data=buffer.getvalue(),
            file_name="hbeam_count_result.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.info("왼쪽에서 사진을 올리고 'AI로 형강 개수 분석하기'를 눌러 주세요.")
        st.markdown(
            '<p class="hint">분석이 끝나면 이곳에 탐지 개수와 '
            '회전 경계 상자(OBB)가 표시됩니다.</p>',
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "개인 학습용 시제품 · 공개 형강 생산라인 이미지 400장 활용 · "
    "검증 이미지 20장 기준 개수 완전 일치 85%, MAE 0.6개. "
    "검증 데이터에 라벨 오류가 발견되었으며 실제 현대제철 공정에서 검증된 성능은 아닙니다."
)
