# CaptchaCracker v2

TensorFlow 2.13.1, uv 패키지 매니저, Docker 기반으로 학습/검증 파이프라인을 새로 구성한 버전입니다. 숫자(0-9)로만 이루어진 200x50 픽셀 PNG 캡차 이미지를 학습 대상으로 하며, 모든 주요 하이퍼파라미터를 환경 변수로 조정할 수 있습니다.

## 주요 특징
- **최신 TensorFlow 2.13.1**: pip index 기준 최신 안정 버전을 사용합니다.
- **uv 기반 의존성 관리**: `pyproject.toml` + `uv.lock` 조합으로 재현 가능한 환경을 제공합니다.
- **Docker + compose 지원**: GPU/CPU 환경을 통일된 방법으로 재현할 수 있게 설계했습니다.
- **환경 변수 중심 구성**: 데이터 경로, 이미지 크기, 추론 길이, 학습률 등 대부분 값을 `.env` 파일에서 제어합니다.
- **문서화**: `documents/getting_started.md`에서 초심자도 따라 할 수 있는 전 과정을 안내합니다.

## 빠른 시작 (로컬)
1. **uv 설치**: 설치되어 있다면 생략합니다. (https://docs.astral.sh/uv/install)  
2. **환경 파일 준비**
   ```bash
   cd CaptchaCracker_v2
   cp .env.example .env
   # 데이터/모델 경로 등 필요한 값 수정
   ```
3. **데이터 배치**: `.env`의 `CC_DATA_DIR`에 PNG 학습 이미지를 넣습니다. 파일명은 정답 숫자 문자열이어야 합니다. (예: `4829.png`)
4. **학습 실행**
   ```bash
   uv run python -m captcha_cracker_v2.train --log-extra
   ```
5. **검증 실행**
   ```bash
   uv run python -m captcha_cracker_v2.validate --sample-count 10
   ```

### 스크립트 사용
```
./scripts/train.sh --epochs 50
./scripts/validate.sh --weights ./model_checkpoints/weights_v2.keras
```

## Docker 워크플로
```bash
cd CaptchaCracker_v2
cp .env.example .env
# (필요 시) docker compose build
docker compose run --rm trainer
# 검증
docker compose run --rm validator --weights /app/model_checkpoints/weights_v2.keras
```
볼륨 마운트로 `data/`, `model_checkpoints/`, `documents/training_logs/`를 호스트와 공유합니다. GPU를 사용하려면 compose 파일에 `runtime: nvidia` 또는 `deploy.resources` 설정을 추가하세요.

## 환경 변수 요약
| 변수 | 설명 | 기본값 |
| --- | --- | --- |
| `CC_DATA_DIR` | 학습/검증 이미지 디렉터리 | `./data` |
| `CC_IMAGE_WIDTH` / `CC_IMAGE_HEIGHT` / `CC_IMAGE_CHANNELS` | 입력 이미지 가로/세로/채널 | `200` / `50` / `1` |
| `CC_MAX_LABEL_LENGTH` | 캡차 최대 길이 | `4` |
| `CC_CHARSET` | 허용 문자 집합 | `0123456789` |
| `CC_BATCH_SIZE` | 배치 크기 | `64` |
| `CC_EPOCHS` | 학습 epoch 수 | `30` |
| `CC_LEARNING_RATE` | Adam 학습률 | `0.001` |
| `CC_VALIDATION_SPLIT` | 검증 데이터 비율 | `0.2` |
| `CC_AUGMENT_ROTATION` | 데이터 증강 회전 각도(도) | `3.0` |
| `CC_MODEL_DIR` / `CC_MODEL_NAME` | 가중치 저장 경로/파일명 | `./model_checkpoints` / `weights_v2` |
| `CC_NUM_WORKERS` | tf.data 병렬 처리 수 | `4` |

추가 변수는 `.env.example`를 참고하세요.

## 디렉터리 구조
```
CaptchaCracker_v2/
├── Dockerfile
├── README.md
├── docker-compose.yml
├── documents/
│   ├── getting_started.md
│   └── training_logs/
├── pyproject.toml
├── uv.lock
├── scripts/
│   ├── train.sh
│   └── validate.sh
├── src/
│   └── captcha_cracker_v2/
│       ├── __init__.py
│       ├── config.py
│       ├── data_pipeline.py
│       ├── model.py
│       ├── predictions.py
│       ├── train.py
│       └── validate.py
└── data/
```

## 문서 & 추가 자료
- `documents/getting_started.md`: uv 설치, 데이터 준비, 학습/검증, Docker 활용, 문제 해결 등을 단계별로 설명합니다.
- 학습 로그 및 TensorBoard 이벤트는 `documents/training_logs/` 아래에 자동 저장됩니다.

## 문제 해결 팁
- **메모리 부족**: `CC_BATCH_SIZE`를 줄이고 `CC_IMAGE_CHANNELS`를 1로 유지합니다.
- **정확도 정체**: 데이터셋 품질을 다시 확인하거나 `CC_MAX_LABEL_LENGTH`, `CC_AUGMENT_ROTATION` 등을 조정합니다.
- **환경 충돌**: `uv run --python 3.11 ...` 처럼 명시적으로 파이썬 버전을 지정할 수 있습니다.

추가 문의 사항은 `documents/getting_started.md`를 먼저 확인하세요. EOF
