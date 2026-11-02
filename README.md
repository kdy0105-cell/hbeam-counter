# H형강 자동 카운팅 웹앱

권도엽님의 공개 H형강 생산라인 이미지 기반 개인 학습 프로젝트입니다.
이미지 한 장을 올리면 학습된 YOLO OBB 모델(`best.pt`)로 보이는 형강의
개수를 계산하고, 회전 경계 상자를 파란색으로 표시합니다.

## 포함 파일

| 파일 | 용도 |
| --- | --- |
| `app.py` | Streamlit 웹페이지와 이미지 분석 코드 |
| `best.pt` | Google Colab에서 학습한 YOLO OBB 가중치 |
| `requirements.txt` | Python 패키지 목록 |
| `packages.txt` | 배포용 시스템 라이브러리 목록 |
| `.streamlit/config.toml` | 웹페이지 테마 |

## 컴퓨터에서 실행하기

Python 3.11 또는 3.12 환경을 권장합니다.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

터미널에 나타나는 `http://localhost:8501` 주소로 접속하세요.
첫 설치에는 PyTorch 등 모델 실행 패키지가 내려받아져 시간이 걸릴 수 있습니다.

## 누구나 링크로 접속하도록 배포하기 (Streamlit Community Cloud)

1. [GitHub](https://github.com/)에서 로그인한 후 새 저장소를 만듭니다.
2. 압축파일을 푼 뒤 **`app.py`, `best.pt`, `requirements.txt`, `packages.txt`** 파일을
   새 GitHub 저장소에 업로드하고 저장(Commit changes)합니다.
   `.streamlit/config.toml`은 선택 사항이며, 넣으면 테마가 그대로 적용됩니다.
3. [Streamlit Community Cloud](https://share.streamlit.io/)에 GitHub 계정으로 로그인합니다.
4. **Create app → Yup, I have an app**을 선택하고 방금 만든 저장소를 고릅니다.
5. **Main file path**에 `app.py`를 입력한 뒤 **Deploy**를 누릅니다.
6. 배포된 `.streamlit.app` 주소로 접속해 사진을 올리고 분석을 실행합니다.

무료 클라우드 서버는 CPU/메모리 제약이 있어 모델 로딩 및 첫 분석이 느릴 수 있습니다.
모델 호환성이나 메모리 문제로 배포에 실패하면 로그를 확인하여 환경을 조정해야 합니다.

## 정확한 설명 및 제한

- 정지 이미지 **한 장에서 보이는 형강 개수**를 추정합니다. 영상 속 움직이는
  H형강을 추적하거나 중복 없는 누적 생산량을 측정하지는 않습니다.
- 검증 이미지 20장에서 개수 정확 일치 85%(17/20), 평균 절대 개수 오차
  0.6개를 확인했습니다. 검증 데이터에 원본 정답 라벨 오류가 한 건 이상
  있었고, 프레임 간 유사도가 높아 실제 현장 신뢰성을 입증하지 않습니다.
- 생산 현장, 품질관리 설비, 현대제철 내부 공정에 배포하거나 적용한 경험은 아닙니다.
- 학습 데이터 출처: Roboflow Universe의 공개 `newHBeam` 데이터셋(제공 당시 CC BY 4.0).
  실제 사례 발표 시 해당 출처를 명시해 주세요.
- 상용 배포나 별도 제공 전에는 Ultralytics 라이선스 적용 범위도 확인하세요.
