# CaptchaCracker v2 가이드

처음으로 ML 기반 캡차 모델을 다루는 사용자를 위해, 환경 구성부터 학습/검증, Docker 활용, 문제 해결까지 단계별로 정리했습니다.

## 1. 준비물 체크리스트
- **운영체제**: macOS, Linux, Windows WSL2 중 하나
- **Python 3.11** 이상 (uv가 자동으로 설치해 줍니다)
- **uv 0.9+**: https://docs.astral.sh/uv/install 를 참고해 설치하세요.
- **GPU 학습(선택)**: NVIDIA 드라이버 + CUDA 12.x, Docker 사용 시 `--gpus all`

## 2. 프로젝트 구조 이해
```
CaptchaCracker_v2
├── data/                    # 학습/검증 PNG 파일 (파일명=정답)
├── documents/getting_started.md
├── documents/training_logs/ # CSV/TensorBoard 로그가 자동 저장
├── model_checkpoints/       # 학습된 가중치(.keras)
├── scripts/*.sh             # uv 명령을 감싼 헬퍼 스크립트
├── src/captcha_cracker_v2   # 파이썬 모듈 (config, 데이터로더, 모델, train/validate)
└── Dockerfile, docker-compose.yml
```

## 3. 환경 변수 구성
1. `.env.example`를 복사해 `.env` 파일을 만듭니다.
2. 필수 수정 항목
   - `CC_DATA_DIR`: 학습 PNG가 위치한 절대/상대 경로
   - `CC_MAX_LABEL_LENGTH`: 캡차 글자 수 (예: 4)
   - `CC_MODEL_DIR`: 가중치를 저장할 디렉터리
3. 추가 옵션
   - `CC_AUGMENT_ROTATION`: 데이터 증강 회전 각도 (도 단위)
   - `CC_BATCH_SIZE`, `CC_EPOCHS`, `CC_LEARNING_RATE`

모든 하이퍼파라미터는 환경 변수를 통해 관리되므로, 설정 변경 후 재학습 시에도 코드 수정이 필요 없습니다.

## 4. 데이터 준비 요령
- PNG 파일명 = 정답 숫자 문자열 (`0001.png`, `4830.png` 등)
- 해상도는 200x50이 권장되지만, 다른 크기를 사용하려면 `CC_IMAGE_WIDTH`, `CC_IMAGE_HEIGHT`를 맞게 변경하세요.
- 회색조 이미지를 추천하며, 컬러 데이터는 `CC_IMAGE_CHANNELS=3`으로 설정하면 됩니다.
- 최소 1,000장 이상의 샘플을 확보해야 안정적인 학습이 가능합니다.

## 5. uv를 통한 로컬 학습
```bash
cd CaptchaCracker_v2
uv run python -m captcha_cracker_v2.train --log-extra
```
- uv는 `pyproject.toml`과 `uv.lock`을 읽어 필요한 패키지를 자동 설치합니다.
- `--log-extra` 옵션은 학습 종료 후 검증 세트의 예측 결과를 표로 보여줍니다.
- 추가 옵션 예시: `./scripts/train.sh --epochs 60 --learning-rate 5e-4`

학습 결과는 기본적으로 `model_checkpoints/weights_v2.keras`에 저장되며, CSV 로그와 TensorBoard 이벤트는 `documents/training_logs/` 하위에 생성됩니다.

## 6. 모델 검증
```bash
./scripts/validate.sh --weights ./model_checkpoints/weights_v2.keras --sample-count 8
```
- `model.evaluate()` 결과(전체 loss 및 각 문자 포지션의 정확도)가 표 형태로 출력됩니다.
- `--weights`를 생략하면 `.env`에 정의된 경로/파일명이 사용됩니다.

## 7. Docker 실행
1. `.env` 파일을 준비합니다.
2. 이미지를 빌드합니다.
   ```bash
   docker compose build
   ```
3. 학습/검증 실행
   ```bash
   docker compose run --rm trainer --epochs 40
   docker compose run --rm validator --sample-count 5
   ```
4. GPU 사용 시 `docker compose run --rm --gpus all trainer ...` 형태로 호출하세요.

## 8. 결과 해석
- `documents/training_logs/history_*.csv`: epoch별 loss/accuracy 추이를 확인할 수 있습니다.
- TensorBoard 사용 시
  ```bash
  tensorboard --logdir documents/training_logs/tensorboard
  ```
- 저장된 `.keras` 가중치는 `uv run python -m captcha_cracker_v2.validate --weights ...` 혹은 별도 추론 스크립트에서 불러 사용할 수 있습니다.

## 9. 자주 묻는 질문
- **데이터가 너무 적어요**: `CC_VALIDATION_SPLIT`을 낮추거나 외부 캡차 수집 스크립트를 작성하세요.
- **학습이 느려요**: 배치 크기를 키우고, `CC_NUM_WORKERS`를 코어 수에 맞게 조정합니다.
- **정확도가 낮아요**: 데이터 품질을 확인하고, `CC_AUGMENT_ROTATION`을 줄이거나 모델 구조(예: `model.py`)에서 Conv 채널 수를 늘리는 방법을 고려하세요.
- **다른 문자 집합이 필요해요**: `.env`의 `CC_CHARSET`과 `CC_MAX_LABEL_LENGTH`를 변경하면 곧바로 적용됩니다.

## 10. 다음 단계
- `train.py`에 커스텀 콜백을 추가하거나, EarlyStopping 조건을 변경해 자신만의 학습 전략을 구축하세요.
- Docker 이미지를 CI 파이프라인에 연결해 자동 학습/검증을 구성할 수 있습니다.
- 추론용 API 또는 CLI를 만들고 싶다면, `predictions.decode_output()`을 활용해 손쉽게 문자열을 복원할 수 있습니다.

필요 시 언제든 `.env` 설정을 변경하고 `uv run ...` 명령을 반복 실행하면 됩니다. 즐거운 실험 되세요!
