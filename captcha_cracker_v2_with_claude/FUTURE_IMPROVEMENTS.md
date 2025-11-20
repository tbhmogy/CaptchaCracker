# CaptchaCracker 향후 개선 계획

## 문서 개요

이 문서는 레거시 마이그레이션(v1.0.0) **이후**에 수행할 개선 사항을 다룹니다.
`REFACTORING_PLAN.md`가 기술 부채 해소에 집중한다면, 본 문서는 **기능 및 성능 향상**에 초점을 맞춥니다.

- **선행 작업**: `REFACTORING_PLAN.md` Phase 1-6 완료
- **대상 버전**: v1.1.0 이상
- **작성일**: 2025-11-19

---

## 개선 카테고리

1. 🎯 **모델 정확도 향상**
2. ⚡ **성능 최적화**
3. 🔧 **모델 아키텍처 개선**
4. 📊 **데이터 및 학습 개선**
5. 🚀 **새로운 기능**
6. 🌐 **프로덕션 준비**
7. 🔬 **연구 및 실험**

---

## 1. 🎯 모델 정확도 향상

### 1.1 데이터 증강 (Data Augmentation)

**현재 상태**: 데이터 증강 없음

**개선 방안**:
```python
# 학습 시 실시간 데이터 증강 적용
augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),           # ±5도 회전
    layers.RandomTranslation(0.1, 0.1),    # 10% 이동
    layers.RandomBrightness(0.2),          # 밝기 조절
    layers.RandomContrast(0.2),            # 대비 조절
    layers.GaussianNoise(0.01),            # 노이즈 추가
])
```

**예상 효과**:
- 과적합(overfitting) 감소
- 다양한 캡챠 스타일에 대한 일반화 성능 향상
- 예상 정확도 향상: 3-5%

**우선순위**: ⭐⭐⭐ High

---

### 1.2 Attention Mechanism 추가

**현재 상태**: Bidirectional LSTM만 사용

**개선 방안**:
```python
# Attention 레이어 추가
attention = layers.Attention()
lstm_output = Bidirectional(LSTM(128, return_sequences=True))(x)
attention_output = attention([lstm_output, lstm_output])
```

또는 Transformer 기반 아키텍처:
```python
# Multi-Head Attention 사용
attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)
attention_output = attention(query=x, key=x, value=x)
```

**예상 효과**:
- 긴 시퀀스에 대한 인식 성능 향상
- 문자 간 관계 학습 개선
- 예상 정확도 향상: 2-4%

**우선순위**: ⭐⭐⭐ High

---

### 1.3 앙상블 (Ensemble) 모델

**개선 방안**:
```python
# 여러 모델의 예측을 결합
models = [
    load_model('weights_v1.h5'),
    load_model('weights_v2.h5'),
    load_model('weights_v3.h5'),
]

# Voting 또는 Averaging
predictions = [model.predict(image) for model in models]
final_prediction = ensemble_vote(predictions)
```

**예상 효과**:
- 예측 안정성 향상
- 예상 정확도 향상: 1-3%
- 단점: 추론 시간 증가 (3배)

**우선순위**: ⭐⭐ Medium

---

### 1.4 CTC Loss 대안 탐색

**현재 상태**: CTC Loss 사용

**대안**:
1. **Attention-based Encoder-Decoder**
   - Seq2Seq with Attention
   - 더 유연한 시퀀스 처리

2. **Transformer Decoder**
   - BERT-style masked prediction
   - 양방향 문맥 활용

**우선순위**: ⭐⭐ Medium (연구 필요)

---

## 2. ⚡ 성능 최적화

### 2.1 모델 경량화

#### 2.1.1 모델 양자화 (Quantization)

**방법**:
```python
# INT8 양자화
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

**예상 효과**:
- 모델 크기: 75% 감소
- 추론 속도: 2-4배 향상
- 정확도 손실: <1%

**우선순위**: ⭐⭐⭐ High

---

#### 2.1.2 지식 증류 (Knowledge Distillation)

**방법**:
```python
# Teacher 모델 (큰 모델)의 지식을 Student 모델 (작은 모델)로 전달
teacher = load_model('large_model.h5')
student = build_small_model()

# Soft targets 학습
student.train_with_teacher(teacher, temperature=3.0)
```

**예상 효과**:
- 모델 크기: 50% 감소
- 추론 속도: 2-3배 향상
- 정확도: 큰 모델의 95%+ 유지

**우선순위**: ⭐⭐ Medium

---

### 2.2 추론 최적화

#### 2.2.1 ONNX 변환

**방법**:
```python
import tf2onnx

# TensorFlow → ONNX 변환
onnx_model = tf2onnx.convert.from_keras(model)

# ONNX Runtime으로 추론
import onnxruntime as ort
session = ort.InferenceSession(onnx_model)
```

**예상 효과**:
- 크로스 플랫폼 호환성
- 추론 속도: 1.5-2배 향상
- 다양한 하드웨어 최적화 (CPU, GPU, TensorRT)

**우선순위**: ⭐⭐⭐ High

---

#### 2.2.2 배치 처리 최적화

**현재 상태**: 이미지를 하나씩 처리

**개선**:
```python
# 동적 배치 크기 지원
def predict_batch(images, batch_size='auto'):
    if batch_size == 'auto':
        # GPU 메모리 기반 자동 배치 크기 결정
        batch_size = estimate_optimal_batch_size()

    results = []
    for batch in batched(images, batch_size):
        results.extend(model.predict(batch))
    return results
```

**예상 효과**:
- 대량 이미지 처리 시 2-5배 속도 향상

**우선순위**: ⭐⭐⭐ High

---

#### 2.2.3 전처리 파이프라인 최적화

**방법**:
```python
# TensorFlow Dataset API 최적화
dataset = tf.data.Dataset.from_tensor_slices(images)
dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

**예상 효과**:
- I/O 병목 제거
- GPU 활용률 향상

**우선순위**: ⭐⭐ Medium

---

### 2.3 Mixed Precision 학습

**방법**:
```python
# Mixed Precision 정책 설정
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# 학습 속도 향상 (GPU 사용 시)
model.compile(optimizer='adam', loss='ctc', metrics=['accuracy'])
```

**예상 효과**:
- 학습 속도: 2-3배 향상
- GPU 메모리 사용량: 50% 감소
- 정확도 손실: 거의 없음

**우선순위**: ⭐⭐⭐ High

---

## 3. 🔧 모델 아키텍처 개선

### 3.1 CNN Backbone 업그레이드

**현재 상태**: 단순한 2-layer CNN

**개선 옵션**:

#### 옵션 A: ResNet Blocks
```python
def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    return ReLU()(x)
```

#### 옵션 B: EfficientNet Backbone
```python
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',  # 또는 None
    input_shape=(50, 200, 1)
)
```

#### 옵션 C: Vision Transformer (ViT)
```python
# Patch embedding + Transformer encoder
patches = extract_patches(image)
encoded = TransformerEncoder()(patches)
```

**우선순위**: ⭐⭐ Medium

---

### 3.2 Bidirectional LSTM 대안

**현재 상태**: Bidirectional LSTM 2층

**대안**:

#### 옵션 A: GRU (더 가벼움)
```python
Bidirectional(GRU(128, return_sequences=True))
Bidirectional(GRU(64, return_sequences=True))
```

#### 옵션 B: 1D Convolution
```python
Conv1D(128, 3, padding='same', activation='relu')
Conv1D(64, 3, padding='same', activation='relu')
```

#### 옵션 C: Transformer Encoder
```python
TransformerEncoder(
    num_layers=2,
    d_model=128,
    num_heads=4,
    dff=512
)
```

**우선순위**: ⭐⭐ Medium

---

### 3.3 Dynamic Image Size 지원

**현재 상태**: 고정 크기 (200x50)만 지원

**개선**:
```python
# 다양한 크기 지원
model = CaptchaModel.load('weights.h5', image_size='auto')

# 추론 시 자동 리사이징
result = model.predict('captcha_300x60.png')  # 자동 조정
```

**구현 방안**:
1. Fully Convolutional Network (FCN) 사용
2. Adaptive Pooling 사용
3. 여러 크기별 모델 학습

**우선순위**: ⭐⭐ Medium

---

## 4. 📊 데이터 및 학습 개선

### 4.1 합성 데이터 생성

**목적**: 학습 데이터 부족 문제 해결

**방법**:
```python
from captcha.image import ImageCaptcha

generator = ImageCaptcha(width=200, height=50)

# 다양한 스타일의 캡챠 생성
for _ in range(10000):
    text = generate_random_text()
    image = generator.generate(text)
    save_image(image, f'{text}.png')
```

**고려사항**:
- 폰트, 색상, 왜곡 정도 다양화
- 실제 캡챠와 유사한 노이즈 추가
- GAN을 활용한 고품질 합성 데이터

**우선순위**: ⭐⭐⭐ High

---

### 4.2 학습 전략 개선

#### 4.2.1 Learning Rate Scheduler
```python
# Cosine Annealing
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000
)

# Warm-up + Cosine Decay
lr_schedule = WarmupCosineDecay(
    warmup_steps=100,
    total_steps=1000,
    initial_lr=1e-5,
    max_lr=1e-3
)
```

#### 4.2.2 Curriculum Learning
```python
# 쉬운 샘플부터 점진적으로 어려운 샘플 학습
epoch_1_10: simple_captchas (4-5 characters, no noise)
epoch_11_30: medium_captchas (5-6 characters, light noise)
epoch_31_100: hard_captchas (6+ characters, heavy noise)
```

#### 4.2.3 Label Smoothing
```python
# One-hot encoding 대신 부드러운 레이블 사용
loss = CategoricalCrossentropy(label_smoothing=0.1)
```

**우선순위**: ⭐⭐ Medium

---

### 4.3 Cross-validation

**현재 상태**: 단순 train/validation split (90/10)

**개선**:
```python
# K-Fold Cross-validation
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
scores = []

for train_idx, val_idx in kf.split(data):
    model = build_model()
    model.fit(data[train_idx], ...)
    score = model.evaluate(data[val_idx])
    scores.append(score)

print(f"Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

**우선순위**: ⭐⭐ Medium

---

### 4.4 Hard Negative Mining

**개념**: 잘못 예측한 샘플을 더 자주 학습

**방법**:
```python
# 예측 오류가 큰 샘플에 가중치 부여
sample_weights = compute_sample_weights(predictions, labels)
model.fit(X, y, sample_weight=sample_weights)
```

**우선순위**: ⭐⭐ Medium

---

## 5. 🚀 새로운 기능

### 5.1 다국어 캡챠 지원

**현재 상태**: 숫자만 지원

**확장**:
- 영문 알파벳 (대소문자)
- 한글
- 혼합 (숫자 + 영문)

**구현**:
```python
# 문자 집합 확장
CHARACTERS = {
    'digits': '0123456789',
    'alpha': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
    'korean': 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ가나다라마바사아자차카타파하...',
    'mixed': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
}

model = CaptchaModel.load('weights.h5', charset='mixed')
```

**우선순위**: ⭐⭐⭐ High

---

### 5.2 신뢰도 기반 재시도

**개념**: 낮은 신뢰도 예측 시 자동 재처리

**구현**:
```python
result = model.predict('captcha.png', return_confidence=True)

if result.confidence < 0.8:
    # 전처리 변경 후 재시도
    result = model.predict(
        'captcha.png',
        preprocessing='aggressive'
    )
```

**우선순위**: ⭐⭐ Medium

---

### 5.3 실시간 비디오 캡챠 인식

**목적**: 웹캠 또는 스크린 캡처에서 실시간 인식

**구현**:
```python
import cv2

cap = cv2.VideoCapture(0)
detector = CaptchaDetector()  # YOLO 기반 캡챠 영역 탐지

while True:
    frame = cap.read()
    captcha_region = detector.detect(frame)
    if captcha_region:
        text = model.predict(captcha_region)
        print(f"Detected: {text}")
```

**우선순위**: ⭐ Low (특수 용도)

---

### 5.4 Web API / REST API

**현재 상태**: Python 라이브러리만 제공

**개선**:
```python
# FastAPI 기반 REST API
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    image = await file.read()
    result = model.predict_bytes(image)
    return {"text": result, "confidence": 0.95}

# 사용 예시
curl -X POST -F "file=@captcha.png" http://localhost:8000/predict
```

**우선순위**: ⭐⭐⭐ High (프로덕션 배포 시)

---

### 5.5 Browser Extension

**목적**: 브라우저에서 캡챠 자동 입력

**구현**:
- Chrome/Firefox Extension 개발
- 페이지의 캡챠 이미지 자동 감지
- 모델 추론 후 자동 입력

**기술 스택**:
- TensorFlow.js (브라우저 내 추론)
- 또는 서버 API 호출

**우선순위**: ⭐ Low (별도 프로젝트)

---

## 6. 🌐 프로덕션 준비

### 6.1 모델 버전 관리

**구현**:
```python
# 모델 레지스트리
from captcha_cracker import ModelRegistry

registry = ModelRegistry('s3://models/')
registry.register(
    model_path='weights_v3.h5',
    version='1.2.0',
    metrics={'accuracy': 0.98, 'speed': '20ms'},
    tags=['production', 'numbers-only']
)

# 프로덕션에서 최신 모델 자동 로드
model = CaptchaModel.load_from_registry(
    stage='production',
    version='latest'
)
```

**도구**: MLflow, DVC, Weights & Biases

**우선순위**: ⭐⭐⭐ High

---

### 6.2 모니터링 및 로깅

**구현**:
```python
# 예측 결과 로깅
import logging
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@prediction_latency.time()
def predict(image):
    result = model.predict(image)
    prediction_counter.inc()
    logger.info(f"Predicted: {result}, confidence: {result.confidence}")
    return result
```

**메트릭**:
- 예측 횟수
- 평균 응답 시간
- 신뢰도 분포
- 오류율

**우선순위**: ⭐⭐⭐ High (프로덕션 시)

---

### 6.3 A/B 테스팅

**목적**: 새 모델과 기존 모델 성능 비교

**구현**:
```python
# Traffic splitting
if random.random() < 0.1:  # 10% 트래픽
    result = model_v2.predict(image)
else:
    result = model_v1.predict(image)

# 메트릭 수집 및 비교
collect_metrics(model_version, result, ground_truth)
```

**우선순위**: ⭐⭐ Medium

---

### 6.4 캐싱

**목적**: 동일 이미지 재예측 방지

**구현**:
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def predict_cached(image_hash: str):
    return model.predict(image_hash)

# 사용
image_hash = hashlib.md5(image_bytes).hexdigest()
result = predict_cached(image_hash)
```

**우선순위**: ⭐⭐ Medium

---

### 6.5 Rate Limiting

**목적**: API 남용 방지

**구현**:
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")  # 분당 10회 제한
async def predict(file: UploadFile):
    ...
```

**우선순위**: ⭐⭐⭐ High (API 제공 시)

---

## 7. 🔬 연구 및 실험

### 7.1 Self-Supervised Learning

**개념**: 레이블 없는 데이터로 사전 학습

**방법**:
```python
# Contrastive Learning (SimCLR)
# 동일 이미지의 다른 augmentation을 가깝게 학습
loss = contrastive_loss(
    embedding1=encoder(augment1(image)),
    embedding2=encoder(augment2(image))
)
```

**예상 효과**:
- 적은 레이블 데이터로 높은 성능
- 일반화 성능 향상

**우선순위**: ⭐ Low (연구 단계)

---

### 7.2 Few-Shot Learning

**개념**: 적은 샘플로 새로운 캡챠 타입 학습

**방법**:
- Meta-Learning (MAML, Prototypical Networks)
- 5-10개 샘플로 새로운 캡챠 스타일 적응

**우선순위**: ⭐ Low (연구 단계)

---

### 7.3 Adversarial Training

**목적**: 적대적 공격에 강건한 모델

**방법**:
```python
# FGSM (Fast Gradient Sign Method)
adversarial_images = generate_adversarial_examples(images)
model.train_on_batch(adversarial_images, labels)
```

**우선순위**: ⭐ Low

---

### 7.4 Neural Architecture Search (NAS)

**목적**: 최적 모델 구조 자동 탐색

**도구**: AutoKeras, KerasTuner

**우선순위**: ⭐ Low (연구 단계)

---

## 우선순위 매트릭스

| 개선 항목 | 영향도 | 구현 난이도 | 우선순위 | 예상 기간 |
|-----------|--------|-------------|----------|-----------|
| 데이터 증강 | High | Low | ⭐⭐⭐ | 1주 |
| ONNX 변환 | High | Medium | ⭐⭐⭐ | 2주 |
| 배치 처리 최적화 | High | Low | ⭐⭐⭐ | 1주 |
| 모델 양자화 | High | Medium | ⭐⭐⭐ | 2주 |
| 합성 데이터 생성 | High | Medium | ⭐⭐⭐ | 2주 |
| 다국어 지원 | Medium | High | ⭐⭐⭐ | 3주 |
| Web API | High | Low | ⭐⭐⭐ | 1주 |
| 모델 버전 관리 | Medium | Medium | ⭐⭐⭐ | 2주 |
| Attention Mechanism | High | High | ⭐⭐⭐ | 3주 |
| Monitoring/Logging | Medium | Low | ⭐⭐⭐ | 1주 |
| Mixed Precision | Medium | Low | ⭐⭐⭐ | 1주 |
| CNN Backbone 업그레이드 | Medium | High | ⭐⭐ | 3주 |
| 앙상블 | Low | Low | ⭐⭐ | 1주 |
| 지식 증류 | Medium | High | ⭐⭐ | 3주 |
| Learning Rate Scheduler | Low | Low | ⭐⭐ | 3일 |

---

## 제안 로드맵

### Phase 1 (v1.1.0) - Quick Wins (2-3주)
**목표**: 빠르게 적용 가능한 개선
- [ ] 데이터 증강
- [ ] 배치 처리 최적화
- [ ] Mixed Precision 학습
- [ ] 합성 데이터 생성
- [ ] Learning Rate Scheduler

**예상 효과**:
- 정확도: 3-5% 향상
- 학습 속도: 2배 향상

---

### Phase 2 (v1.2.0) - 성능 최적화 (3-4주)
**목표**: 추론 속도 및 경량화
- [ ] 모델 양자화
- [ ] ONNX 변환
- [ ] 전처리 파이프라인 최적화
- [ ] 모델 버전 관리

**예상 효과**:
- 추론 속도: 2-4배 향상
- 모델 크기: 75% 감소

---

### Phase 3 (v1.3.0) - 아키텍처 개선 (4-6주)
**목표**: 모델 구조 현대화
- [ ] Attention Mechanism 추가
- [ ] CNN Backbone 업그레이드 (ResNet 또는 EfficientNet)
- [ ] Dynamic Image Size 지원

**예상 효과**:
- 정확도: 5-7% 향상
- 다양한 캡챠 타입 지원

---

### Phase 4 (v1.4.0) - 프로덕션 준비 (2-3주)
**목표**: 실제 서비스 배포
- [ ] Web API (FastAPI)
- [ ] 모니터링 및 로깅
- [ ] Rate Limiting
- [ ] 캐싱
- [ ] A/B 테스팅

**예상 효과**:
- 프로덕션 준비 완료
- 안정적인 서비스 운영

---

### Phase 5 (v2.0.0) - 다국어 및 고급 기능 (6-8주)
**목표**: 기능 확장
- [ ] 다국어 캡챠 지원
- [ ] 앙상블 모델
- [ ] 신뢰도 기반 재시도
- [ ] 지식 증류

**예상 효과**:
- 영문, 한글 등 다양한 캡챠 지원
- 최고 수준의 정확도

---

## 실험 추적

### 실험 템플릿
```markdown
## 실험 #XX: [제목]

**날짜**: YYYY-MM-DD
**목적**: [실험 목적]
**가설**: [검증하려는 가설]

### 실험 설정
- 데이터셋: [사용한 데이터]
- 모델: [모델 아키텍처]
- 하이퍼파라미터:
  - Learning rate: 1e-3
  - Batch size: 16
  - Epochs: 100

### 결과
- Baseline 정확도: 92.5%
- 개선 후 정확도: 95.3% (+2.8%)
- 추론 속도: 15ms → 12ms (20% 향상)

### 분석
[결과 분석 및 인사이트]

### 결론
- [ ] 프로덕션 적용
- [ ] 추가 실험 필요
- [ ] 기각

### 다음 단계
[후속 실험 또는 개선 방향]
```

---

## 성능 벤치마크 (목표)

| 메트릭 | 현재 (v1.0) | 목표 (v2.0) | 개선율 |
|--------|-------------|-------------|--------|
| 정확도 | 92% | 97%+ | +5%p |
| 추론 속도 (CPU) | 50ms | 15ms | 3.3배 |
| 추론 속도 (GPU) | 10ms | 3ms | 3.3배 |
| 모델 크기 | 20MB | 5MB | 75% 감소 |
| 메모리 사용량 | 500MB | 200MB | 60% 감소 |

---

## 참고 자료

### 논문
- **Attention**: "Attention Is All You Need" (Vaswani et al., 2017)
- **CTC**: "Connectionist Temporal Classification" (Graves et al., 2006)
- **Data Augmentation**: "A survey on Image Data Augmentation for Deep Learning" (Shorten & Khoshgoftaar, 2019)
- **Knowledge Distillation**: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

### 블로그 & 튜토리얼
- TensorFlow Model Optimization Toolkit
- ONNX Runtime Best Practices
- Mixed Precision Training Guide

### 관련 프로젝트
- Tesseract OCR
- EasyOCR
- PaddleOCR

---

## 기여 가이드

커뮤니티 기여를 환영합니다! 다음 항목에 기여할 수 있습니다:

1. **새로운 개선 아이디어 제안**
   - GitHub Issue에 제안서 작성
   - 실험 결과 공유

2. **실험 결과 제출**
   - 위 실험 템플릿 사용
   - Pull Request로 문서 업데이트

3. **새로운 캡챠 데이터셋 공유**
   - 다양한 스타일의 캡챠 이미지
   - 레이블링된 데이터

---

**문서 버전**: 1.0
**최종 수정**: 2025-11-19
**상태**: 🟢 검토 중 (Backlog)

**Note**: 본 문서는 살아있는 문서(Living Document)입니다.
실험 결과와 커뮤니티 피드백에 따라 지속적으로 업데이트됩니다.
