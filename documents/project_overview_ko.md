# CaptchaCracker 개요 (주니어 개발자용)

## 프로젝트 한눈에 보기
- 목표: 숫자 캡차 이미지를 인식해 문자열(예: `023062`)을 뽑아내는 딥러닝 모델을 쉽게 학습·적용하는 파이썬 라이브러리.
- 핵심 모듈: `CaptchaCracker/core.py`에서 모델 생성(`CreateModel`)과 추론(`ApplyModel`) 로직 제공, `train_model.py`로 학습, `download_captcha.py`로 캡차 다운로드 및 예측.
- 기본 데이터 규칙: `data/train_numbers_only*/` 아래 PNG 파일 이름이 정답 라벨(예: `123456.png`). 이미지 크기 기본값 200x50, 라벨 길이 6자리 숫자.
- 기본 가중치: 학습 후 `model/weights_v2.h5`에 저장(배포용 가중치는 `model/`에 포함되어 있음).

## 모델 설계 흐름

1) **입력 전처리** (`core.py:118-133`)
   - PNG → 흑백 변환 → `float32` 스케일링(0~1) → 200x50 리사이즈.
   - `tf.transpose`로 `(너비, 높이, 채널)` 형태로 바꿔 너비 방향을 "시간 축"으로 사용.
   - 왜? LSTM은 시퀀스를 처리하므로 좌→우 방향(너비)이 시간이어야 함.

2) **특징 추출(CNN)** (`core.py:142-162`)
   - `Conv2D(32, he_normal, same padding)` + `MaxPooling(2x2)`
   - `Conv2D(64, he_normal, same padding)` + `MaxPooling(2x2)`
   - 결과: 공간 해상도 1/4로 축소 (200x50 → 50x12), 64개 특징 맵 추출.

3) **시퀀스 변환** (`core.py:168-171`)
   - `Reshape`: (50, 12, 64) → (50, 768) 형태로 펼침. 50개 타임스텝, 각각 768차원 특징.
   - `Dense(64, relu)` + `Dropout(0.2)`: 특징 정제 및 과적합 방지.

4) **시퀀스 학습(RNN)** (`core.py:174-175`)
   - `Bidirectional LSTM(128, dropout=0.25, return_sequences=True)`: 양방향으로 문맥 학습.
   - `Bidirectional LSTM(64, dropout=0.25, return_sequences=True)`: 고차원 시퀀스 패턴 학습.
   - 모든 타임스텝 출력 유지 (return_sequences=True) - CTC에 필수!

5) **문자 확률 예측** (`core.py:178`)
   - `Dense(문자수+1, softmax)`: 각 타임스텝에서 문자별 확률 출력.
   - +1은 CTC blank 토큰용.

6) **CTC 손실 및 디코딩** (`core.py:13-32, 180-181, 344-346`)
   - 학습: `CTCLayer`가 정답 문자열 길이와 이미지 폭이 달라도 정렬 없이 손실 계산.
   - 추론: `ctc_decode(greedy=True)`로 최종 문자열 복원.
   - StringLookup의 +1 오프셋으로 blank 제거 후 문자 변환.

## 실제 코드 따라가기 (핵심 플로우)
- 학습 진입점: `train_model.py` → `CreateModel(train_img_path_list, img_width, img_height)` → `train_model(epochs=100)` → 학습된 가중치가 `model/weights_v2.h5`에 저장.
- 데이터 로더: `CreateModel.encode_single_sample`가 PNG를 읽고 흑백 변환 → 리사이즈 → `tf.transpose` 후 라벨을 숫자 인덱스로 변환(`StringLookup`).
- 모델 구성: `CreateModel.build_model`이 CNN(Conv+Pool) → `Reshape` → BiLSTM 2단 → `Dense(softmax)` → `CTCLayer`로 학습용 모델을 만듦.
- 추론 경로: `ApplyModel.__init__`에서 동일한 구조를 만들고 `load_weights`로 가중치 로드 → `prediction_model`을 image 입력에서 `dense2` 출력까지만 잘라서 사용.
- 예측 호출: `ApplyModel.predict`/`predict_from_bytes`가 전처리 후 `prediction_model.predict` → `decode_batch_predictions`로 문자 시퀀스 복원(CTC 디코딩).

## 이미지가 문자열이 되는 과정 (비전공자용 쉬운 설명)

### 단계별 상세 과정

**1) 이미지 읽기 & 정규화**
- PNG 파일을 읽어서 흑백(1채널)으로 변환
- 픽셀 값을 0~1 사이로 스케일링 (원래는 0~255)
- 왜? 신경망은 작은 숫자로 계산해야 안정적

**2) 크기 맞추기 & 회전**
- 200x50 픽셀로 리사이즈 (모든 이미지 동일 크기로)
- **전치(Transpose)**: (높이50, 너비200) → (너비200, 높이50)
- 왜? LSTM이 좌→우로 읽을 수 있게 너비를 "시간 축"으로 만듦
- 비유: 책을 읽듯이 왼쪽부터 오른쪽으로 순서대로 보기

**3) 특징 뽑기 (CNN의 역할)**
- 1단계 Conv: 32개 필터로 "선, 곡선, 모서리" 같은 저수준 특징 추출
- 1단계 Pooling: 크기를 절반으로 축소 (200x50 → 100x25)
- 2단계 Conv: 64개 필터로 "숫자 모양" 같은 고수준 특징 추출
- 2단계 Pooling: 다시 절반으로 축소 (100x25 → 50x12)
- 결과: 원본 이미지를 "숫자 모양"이라는 의미 있는 특징으로 압축

**4) 시퀀스로 변환 (Reshape)**
- (50, 12, 64)를 (50, 768)로 펼침
- 의미: 50개 위치, 각 위치마다 768개 특징 값
- 비유: 이미지를 50개의 "타임스텝"으로 나눔 (왼쪽부터 오른쪽까지)

**5) 순서 이해 (BiLSTM의 역할)**
- **첫 번째 BiLSTM(128)**: 문자들 사이의 관계 학습
  - 좌→우 LSTM: "앞에 뭐가 있었지?"
  - 우→좌 LSTM: "뒤에 뭐가 오지?"
  - 합치기: 양쪽 정보로 현재 위치 더 정확히 판단
- **두 번째 BiLSTM(64)**: 더 복잡한 패턴 학습
  - 예: "2 다음에 3이 올 확률이 높다" 같은 순서 규칙

**6) 글자 확률 계산 (Dense Layer)**
- 각 타임스텝(위치)마다 문자별 확률 출력
- 예: 위치 10에서 → [0.05, 0.02, 0.85, 0.03, ...] (2번 인덱스가 0.85로 가장 높음)
- 결과: 50개 위치 × (숫자 10개 + blank 1개) = (50, 11) 확률 행렬

**7) 최종 문자열 복원 (CTC Decoding)**
- **Greedy Search**: 각 위치에서 가장 높은 확률의 문자 선택
- **중복 제거**: "2-22-55-8" → "2-2-5-8"
- **Blank 제거**: "2-2-5-8" → "258"
- **+1 오프셋 적용**: 인덱스를 실제 문자로 변환
- 최종 결과: "258" 문자열 출력!

### 전체 흐름 요약
```
PNG 이미지 (200x50)
  ↓ 전처리
Tensor (200, 50, 1) - 시간축이 너비 방향
  ↓ CNN
특징 맵 (50, 12, 64) - 4배 축소
  ↓ Reshape
시퀀스 (50, 768) - 50개 타임스텝
  ↓ BiLSTM
문맥 이해 (50, 256) - 양방향 학습
  ↓ Dense
확률 분포 (50, 11) - 각 위치별 문자 확률
  ↓ CTC Decode
최종 문자열 "023062"
```

## 폴더 & 주요 파일
- `CaptchaCracker/core.py`: 모델 정의, 전처리, CTC 디코딩.
- `train_model.py`: 예제 학습 스크립트(기본 100 epochs, 데이터는 `data/train_numbers_only*/`).
- `download_captcha.py`: URL에서 캡차를 내려받아 예측 후 파일명을 예측값으로 변경.
- `assets/`: README에 쓰이는 예시 이미지.
- `model/`: 배포용 가중치(`weights_v2.h5` 등).

## 빠른 실행 절차
```bash
# 1) 의존성 설치 (권장: 가상환경)
python -m pip install -r requirements.txt

# 2) 로컬 수정 반영을 위해 편집 설치
pip install -e .

# 3) 학습 실행 (데이터 준비 필요)
python train_model.py
# 완료 후 model/weights_v2.h5 생성 또는 갱신

# 4) 예측(스크립트)
python download_captcha.py

# 5) 예측(코드 직접 사용)
python - <<'PY'
import CaptchaCracker as cc
AM = cc.ApplyModel("model/weights_v2.h5", img_width=200, img_height=50, max_length=6, characters=set("0123456789"))
print(AM.predict("data/target.png"))
PY
```

## 데이터 준비 팁
- 파일명=정답: `123456.png`처럼 숫자 문자열만 포함해야 함. 잘못된 철자가 있으면 학습이 흔들림.
- 클래스 집합: 기본은 숫자 `0-9`. 문자나 기호를 추가하면 `CreateModel`과 `ApplyModel`의 `characters` 인자, 그리고 `max_length`를 함께 조정.
- 분할 비율: 내부적으로 90% 학습 / 10% 검증으로 나뉨(`split_data`). 필요 시 함수를 수정해 비율을 바꿀 수 있음.
- 이미지 크기: 학습·추론 모두 동일해야 함. 변경 시 `img_width`, `img_height`를 동일하게 전달하고 가중치도 새로 학습해야 함.

## 튜닝 포인트

### 주요 하이퍼파라미터 (코드 위치 참고)

**학습 제어**
- `epochs`: 기본 100 (`train_model.py:17`). 과적합 시 줄이거나 `earlystopping=True`로 조기 종료 사용.
- `early_stopping_patience`: 10 에포크 고정 (`core.py:92`). 검증 손실이 10번 동안 개선되지 않으면 학습 종료.
- `batch_size`: **16으로 하드코딩** (`core.py:62`). 변경하려면 해당 줄을 직접 수정 필요. GPU 메모리 부족 시 8 이하로 줄이기.

**최적화 설정**
- `optimizer`: **Adam 사용** (`core.py:188`). Learning rate는 기본값 (약 0.001).
- `dropout`: Dense 레이어 후 **0.2** (`core.py:171`), 각 LSTM 레이어마다 **0.25** (`core.py:174-175`). 과적합 방지용.

**모델 구조**
- `kernel_initializer`: Conv2D 레이어에 **"he_normal"** 사용 (`core.py:147, 158`). ReLU 활성화 함수에 적합한 가중치 초기화.
- `padding`: Conv2D에 **"same"** 적용 (`core.py:148, 159`). 입력과 출력 크기 동일하게 유지.
- `return_sequences=True`: LSTM이 **모든 타임스텝의 출력** 반환 (`core.py:174-175`). CTC 디코딩에 필수.

**데이터 관련**
- 라벨 길이: 다른 길이의 캡차를 쓴다면 `max_length`를 변경하고 데이터 파일명도 그 길이에 맞추어 생성.
- 다운샘플 비율: CNN 풀링 단계(2회)로 4배 축소 (`core.py:64`). 텍스트가 너무 뭉개지면 풀링 횟수를 줄이고 `downsample_factor`도 조정.

**데이터 파이프라인 최적화** (`core.py:72-75`)
- `num_parallel_calls=AUTOTUNE`: 전처리 병렬화로 학습 속도 향상.
- `prefetch(buffer_size=AUTOTUNE)`: GPU가 학습하는 동안 다음 배치 미리 준비.

## 용어 미니 사전 (코드 연결)

### 기본 용어
- **CNN(합성곱 신경망)**: `Conv2D` 블록이 이미지 패턴을 추출.
- **MaxPooling**: `MaxPooling2D`가 특징 맵 크기를 줄여 계산량 감소·잡음 제거.
- **BiLSTM**: `Bidirectional(LSTM)`이 양방향으로 문맥을 읽어 글자 경계를 더 잘 잡음.
  - 좌→우 방향: 앞 문자를 보고 현재 문자 예측
  - 우→좌 방향: 뒤 문자를 보고 현재 문자 예측
  - 두 방향 정보를 합쳐서 더 정확한 문자 인식

### 핵심 개념 상세 설명

**Transpose (전치)의 필요성** (`core.py:129`)
```
왜 필요한가?
- LSTM은 시퀀스 데이터를 (시간, 특징) 형태로 처리
- 이미지는 (높이, 너비, 채널) 형태
- 문자는 좌→우로 배열되므로 "너비"가 시간 축이어야 함

실제 변환:
Before: (높이50, 너비200, 채널1) - 이미지 형태
After:  (너비200, 높이50, 채널1) - 시퀀스 형태
결과: LSTM이 이미지의 왼쪽부터 오른쪽으로 순차적으로 읽음
```

**CTC (Connectionist Temporal Classification)** (`core.py:13-32`, `core.py:180-181`)
```
일반 분류와의 차이:
- 일반: 각 문자의 "정확한 위치"를 미리 알아야 함
- CTC: 문자가 "어디서 시작하는지" 몰라도 학습 가능

동작 원리 예시:
모델 출력: "2-22--5-555--8-88-" (블랭크를 -로 표시)
  ↓ 중복 제거
         "2-2-5-5-8-8-"
  ↓ 블랭크(-) 제거
         "258"

실제 코드: core.py:344-346에서 ctc_decode로 처리
```

**Greedy Search** (`core.py:344`)
```
동작 방식:
- 각 시점에서 가장 확률 높은 문자만 선택
- 예: [0.1, 0.7, 0.2] → 2번째 문자 선택 (확률 0.7)

장단점:
✅ 빠름: O(n) 시간 복잡도
❌ 최적해 보장 안 됨: 전체적으로는 더 좋은 조합이 있을 수 있음

대안: Beam Search (현재 미구현)
- 여러 후보를 동시에 고려
- 더 정확하지만 느림
```

**Reshape의 차원 변환** (`core.py:168-169`)
```
목적: CNN의 2D 출력을 LSTM이 처리할 수 있는 1D 시퀀스로 변환

실제 계산 (200x50 이미지 기준):
1. CNN 출력: (너비/4, 높이/4, 필터64) = (50, 12, 64)
2. Reshape: (50, 12×64) = (50, 768)
   - 50 타임스텝
   - 각 타임스텝마다 768개 특징

코드: new_shape = ((img_width//4), (img_height//4) * 64)
```

**StringLookup과 인덱스 오프셋** (`core.py:52-58`, `core.py:350`)
```
역할: 문자 ↔ 숫자 양방향 변환 테이블

예시 vocabulary: ['', '0', '1', '2', '3', ..., '9']
- 0번 인덱스: 빈 문자 (CTC blank용)
- 1번 인덱스: '0'
- 2번 인덱스: '1'
...

주의사항:
- 디코딩 시 +1 오프셋 필요 (core.py:350)
- num_to_char(res+1): CTC blank(0번)를 건너뛰고 실제 문자 추출
- 학습과 추론에서 동일한 vocabulary 순서 필수
```

### 파라미터 설명
- `characters`: `StringLookup`의 vocabulary로 쓰이며, 모델이 예측할 수 있는 문자 목록.
- `max_length`: `decode_batch_predictions`에서 자르는 최대 길이. 실제 캡차 길이 이상이어야 함.

## 자주 겪는 이슈 & 해결법

### 1. Characters 불일치 오류

**문제 상황**: 학습 시와 추론 시 문자 집합이 다름
```python
# ❌ 잘못된 예 - 학습 때는 영문 포함, 추론 때는 숫자만
# 학습 시
CM = CreateModel(images)  # 자동 추출: {'0-9', 'A-Z'}
CM.train_model()

# 추론 시
AM = ApplyModel("weights.h5", characters=set('0123456789'))  # 숫자만!
AM.predict("test.png")  # 'A' 예측 시 오류 발생!
```

**해결 방법**:
```python
# ✅ 올바른 예 - 학습과 동일한 문자 집합 사용
AM = ApplyModel(
    "weights.h5",
    characters=set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # 학습과 동일
)
```

**핵심**: `ApplyModel`의 `characters`는 반드시 학습 시 사용한 문자 집합과 **완전히 동일**해야 함!

### 2. Max Length 불일치 문제

**문제 상황**: 학습 데이터보다 짧은 max_length 설정
```python
# ❌ 잘못된 예
# 학습 데이터: 6자리 숫자 (예: 023062.png)
AM = ApplyModel("weights.h5", max_length=4)  # 4로 설정!
result = AM.predict("test.png")  # 결과: "0230" (앞 4자리만)
# 실제 정답: "023062" → 뒤 2자리 손실!
```

**해결 방법**:
```python
# ✅ 올바른 예
AM = ApplyModel("weights.h5", max_length=6)  # 학습 데이터와 동일
```

**핵심**: `max_length`는 학습 데이터의 최대 라벨 길이 **이상**으로 설정!

### 3. 이미지 크기 불일치 오류

**문제 상황**: 학습과 추론에서 다른 이미지 크기 사용
```python
# ❌ 잘못된 예
# 학습 시: 200x50
CM = CreateModel(images, img_width=200, img_height=50)

# 추론 시: 300x50 (다른 크기!)
AM = ApplyModel("weights.h5", img_width=300, img_height=50)
# → 가중치 shape 불일치로 오류 발생!
```

**해결 방법**:
```python
# ✅ 올바른 예 - 학습과 동일한 크기
AM = ApplyModel("weights.h5", img_width=200, img_height=50)
```

또는 다른 크기로 변경이 필요하면:
```python
# 새 크기로 재학습 필요
CM = CreateModel(images, img_width=300, img_height=50)
CM.train_model()  # 새 가중치 생성
```

**핵심**: 이미지 크기 변경 = **반드시 재학습** 필요!

### 4. Batch Size 조정 방법

**문제 상황**: GPU 메모리 부족 (OOM 오류)
```bash
# 오류 메시지 예시
ResourceExhaustedError: OOM when allocating tensor...
```

**해결 방법**: `core.py` 직접 수정 필요
```python
# core.py:62 수정
batch_size = 16  # 기본값

# GPU 메모리 부족 시
batch_size = 8   # 또는 4로 줄이기

# GPU 메모리 여유 시
batch_size = 32  # 또는 64로 늘리기
```

**주의**: 현재 코드는 batch_size가 **하드코딩**되어 있어 파라미터로 전달 불가!

### 5. 예측 결과가 이상한 경우

**증상**: 글자가 밀리거나 끊김, 완전히 다른 결과

**점검 항목**:
1. **파일명(라벨) 확인**: `123456.png`처럼 정확한 라벨인가?
   ```bash
   # 잘못된 예: image_001.png, captcha_1.png
   # 올바른 예: 123456.png, 987654.png
   ```

2. **데이터 품질**: 학습 데이터와 테스트 이미지의 노이즈/왜곡 정도 비슷한가?

3. **Epochs 부족**: 학습이 충분히 진행되었나?
   ```python
   # 학습 로그에서 loss 확인
   # loss가 계속 감소하는 중이면 epochs 늘리기
   CM.train_model(epochs=200)  # 기본 100에서 증가
   ```

4. **Characters 확인**: 새로운 문자가 추가되었는데 재학습 안 했나?
   ```python
   # 문자 추가 시 반드시 재학습!
   CM = CreateModel(images)  # 새 문자 포함된 데이터
   CM.train_model()
   CM.model.save_weights("model/weights_new.h5")
   ```

## 다음 단계 아이디어

### 정확도 개선
- **데이터 늘리기**: 다양한 배경·폰트로 합성 데이터를 추가해 일반화 향상.
- **Data Augmentation**: 회전, 노이즈, 왜곡 등을 추가해 강건성 향상.
- **Beam Search 구현**: Greedy Search 대신 Beam Search로 디코딩 정확도 개선 (느리지만 더 정확).

### 모니터링 & 평가
- **품질 확인**: 학습 후 `download_captcha.py`나 별도의 스크립트로 예측 결과를 여러 샘플에 대해 출력·비교해 정확도 추적.
- **TensorBoard 연동**: 학습 곡선, 가중치 분포 시각화로 모델 동작 이해.
- **검증 세트 정확도 측정**: 학습 중 정확도 계산해 과적합 여부 판단.

### 성능 최적화
- **경량화**: 추론 속도가 중요하면 LSTM 유닛 수를 줄이거나 양자화(TensorFlow Lite)를 검토.
- **Mixed Precision**: GPU 메모리 절약 및 학습 속도 향상 (fp16).
- **Model Pruning**: 불필요한 가중치 제거로 모델 크기 축소.

### 코드 개선
- **Batch Size 파라미터화**: `core.py:62`의 하드코딩된 값을 함수 인자로 변경.
- **중복 코드 제거**: `CreateModel.build_model`과 `ApplyModel.build_model`을 공통 함수로 추출.
- **자동 테스트**: 학습/추론 파이프라인에 대한 단위 테스트 추가.

## 고급 팁 & 참고사항

### 모델 재현성 확보
학습과 추론 시 다음 항목들이 **완전히 일치**해야 합니다:
- `img_width`, `img_height`: 이미지 크기
- `max_length`: 라벨 최대 길이
- `characters`: 문자 집합 (순서도 동일해야 함!)
- `downsample_factor`: CNN 다운샘플링 비율 (모델 구조 변경 시)

불일치 시 오류 발생하거나 예측 결과가 이상해집니다!

### 학습 데이터 준비 체크리스트
✅ 파일명이 정답 라벨과 정확히 일치 (예: `123456.png`)
✅ 모든 이미지가 동일한 포맷 (PNG)
✅ 라벨 길이가 `max_length` 이하
✅ 모든 문자가 `characters` 집합에 포함됨
✅ 최소 1000장 이상 권장 (딥러닝 특성상 데이터가 많을수록 좋음)
✅ 다양한 변형(노이즈, 왜곡) 포함으로 일반화 능력 향상

### TensorFlow 버전 주의사항
- 현재 프로젝트는 **TensorFlow 2.5.0** 고정
- `layers.experimental.preprocessing.StringLookup` 사용 (deprecated)
- TF 2.8+ 마이그레이션 시 `tf.keras.layers.StringLookup`으로 변경 필요
- Pillow도 9.5.0으로 고정 (호환성)

### 가중치 파일 관리
- `model/weights_v2.h5`: 현재 배포용 가중치
- 실험 중에는 별도 디렉토리에 저장 권장 (예: `experiments/2024-01-01_weights.h5`)
- 가중치 파일은 약 1.7MB (Git에 포함 가능한 크기)
- 버전 관리: 파일명에 날짜나 버전 번호 포함 권장

### 디버깅 팁
```python
# 1. 모델 구조 확인
CM = CreateModel(images)
model = CM.build_model()
model.summary()  # 레이어별 파라미터 수 확인

# 2. 전처리 결과 확인
sample = CM.encode_single_sample(images[0], labels[0])
print(sample['image'].shape)  # (200, 50, 1) 확인
print(sample['label'])  # 인덱스 변환 결과

# 3. 예측 확률 확인 (디버깅용)
AM = ApplyModel("weights.h5", ...)
pred = AM.prediction_model.predict(preprocessed_image)
print(pred.shape)  # (1, 50, 문자수+1) 확인
print(pred[0])  # 각 타임스텝별 확률 분포
```
