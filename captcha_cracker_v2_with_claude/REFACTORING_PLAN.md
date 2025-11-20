# CaptchaCracker 리팩토링 계획서

## 프로젝트 개요
- **프로젝트명**: CaptchaCracker
- **현재 버전**: 0.0.7
- **목표 버전**: 1.0.0
- **목적**: 레거시 코드를 최신 버전으로 마이그레이션
- **작성일**: 2025-11-19

## Executive Summary

이 문서는 CaptchaCracker 프로젝트를 현대적인 기술 스택으로 마이그레이션하기 위한 계획을 담고 있습니다. 본 리팩토링의 주요 목표는 **레거시 API를 최신 버전으로 업데이트**하는 것이며, 모델의 정확도나 로직 개선은 별도 문서(`FUTURE_IMPROVEMENTS.md`)에서 다룹니다.

## 현재 상태 분석

### 기술 스택 (As-Is)
| 항목 | 현재 버전 | 상태 | 비고 |
|------|-----------|------|------|
| Python | 3.8.13 | ❌ EOL | 2024년 10월 지원 종료 |
| TensorFlow | 2.5.0 | ❌ Outdated | 2021년 5월 릴리스 (3년 경과) |
| Keras | 2.5.0.dev | ❌ Deprecated | Nightly 개발 버전 사용 |
| NumPy | 1.19.5 | ❌ Outdated | 2020년 12월 릴리스 |
| Pillow | 9.5.0 | ⚠️ Vulnerable | 알려진 보안 취약점 존재 |

### 주요 레거시 이슈

#### 🔴 심각도: Critical
1. **Deprecated TensorFlow APIs**
   ```python
   # ❌ 현재 (Deprecated)
   from tensorflow.keras import layers
   layers.experimental.preprocessing.StringLookup

   # ✅ 마이그레이션 후
   layers.StringLookup
   ```
   - 📍 위치: `CaptchaCracker/core.py:53, 57, 224, 228`
   - 📅 Deprecated: TensorFlow 2.6+

2. **Deprecated tf.data API**
   ```python
   # ❌ 현재
   tf.data.experimental.AUTOTUNE

   # ✅ 마이그레이션 후
   tf.data.AUTOTUNE
   ```
   - 📍 위치: `CaptchaCracker/core.py:73, 76, 82, 85`
   - 📅 Deprecated: TensorFlow 2.4+

3. **Legacy Keras Backend Functions**
   ```python
   # ❌ 현재
   keras.backend.ctc_batch_cost
   keras.backend.ctc_decode

   # ✅ 마이그레이션 후
   tf.nn.ctc_loss
   tf.nn.ctc_greedy_decoder
   ```
   - 📍 위치: `CaptchaCracker/core.py:17, 344`
   - 📅 Deprecated: Keras 2.x 레거시

4. **Python 3.8 EOL**
   - Python 3.8은 2024년 10월 31일부로 보안 업데이트 중단
   - 최신 TensorFlow 2.18은 Python 3.9-3.12 지원

#### 🟡 심각도: Medium

5. **코드 중복 (Code Duplication)**
   - `build_model()` 메서드가 `CreateModel`과 `ApplyModel` 클래스에 중복 (각 57줄, 총 114줄)
   - 📍 위치: `core.py:136-192`, `core.py:267-323`

6. **Dead Code**
   - `ApplyModel.split_data()` 메서드가 정의되었으나 사용되지 않음
   - 📍 위치: `core.py:325-337`

7. **Hard-coded Values**
   ```python
   batch_size = 16  # 하드코딩
   downsample_factor = 4  # 하드코딩
   dropout_rate = 0.2, 0.25  # 하드코딩
   ```

8. **Type Hints 부재**
   - 전체 코드베이스에 타입 어노테이션 없음
   - IDE 자동완성 및 정적 타입 검사 불가

#### 🟢 심각도: Low

9. **Docstring 부재**
   - 클래스, 메서드에 문서화 문자열 없음
   - API 사용법을 코드에서 파악하기 어려움

10. **테스트 코드 부재**
    - 단위 테스트, 통합 테스트 없음
    - 리팩토링 시 회귀 검증 불가

11. **로깅 시스템 부재**
    - `print()` 문만 사용
    - 프로덕션 환경에서 디버깅 어려움

### 프로젝트 구조 (현재)
```
CaptchaCracker/
├── CaptchaCracker/           # 메인 패키지
│   ├── __init__.py          # v0.0.7
│   └── core.py              # 352 lines
├── model/                    # 학습된 모델 가중치
│   ├── weights.h5           # 원본 가중치
│   └── weights_v2.h5        # 업데이트된 가중치
├── data.zip                  # 학습 데이터 (7.6 MB)
├── train_model.py           # 학습 스크립트
├── download_captcha.py      # 추론 유틸리티
├── setup.py                 # 패키지 설정
├── requirements.txt         # 최소 의존성
└── README.md                # 문서
```

## 확정된 마이그레이션 방향

### ✅ 결정 사항

| 항목 | 선택 | 이유 |
|------|------|------|
| **Python 버전** | 3.12 | 최신 안정 버전, 성능 개선 (~25%) |
| **프레임워크** | TensorFlow 2.18 | 기존 가중치 호환성, 마이그레이션 비용 최소화 |
| **API 설계** | 새로운 설계 | 모던하고 유연한 API, 하위 호환성 미보장 |
| **범위** | 레거시 마이그레이션 | 정확도/로직 개선은 Phase 2 (별도 문서) |

### 기술 스택 (To-Be)

| 항목 | 버전 | 변경 사항 |
|------|------|-----------|
| Python | 3.12.x | 3.8.13 → 3.12 (major upgrade) |
| TensorFlow | 2.18.x | 2.5.0 → 2.18.0 (13개 버전 업그레이드) |
| NumPy | 1.26.x | 1.19.5 → 1.26.x |
| Pillow | 10.4.x | 9.5.0 → 10.4.0 (보안 패치) |

## 새로운 API 설계

### 현재 API (v0.0.7)
```python
from CaptchaCracker import ApplyModel

# 모델 로드
model = ApplyModel('model/weights_v2.h5', (200, 50))

# 예측
result = model.predict('captcha.png')
result = model.predict_from_bytes(image_bytes)
```

**문제점:**
- 클래스명이 직관적이지 않음 (`ApplyModel`)
- 생성자에서 파일 경로와 크기를 동시에 받음
- 컨텍스트 매니저 미지원
- 배치 예측 불가
- 에러 핸들링 부족

### 새로운 API (v1.0.0)

#### 기본 사용법
```python
from captcha_cracker import CaptchaModel

# 1. 모델 로드 (간단한 방법)
model = CaptchaModel.load('model/weights_v2.h5')
result = model.predict('captcha.png')

# 2. 컨텍스트 매니저 사용 (권장)
with CaptchaModel.load('model/weights_v2.h5') as model:
    result = model.predict('captcha.png')
```

#### 고급 사용법
```python
# 3. 설정 옵션
model = CaptchaModel.load(
    'model/weights_v2.h5',
    image_size=(200, 50),        # 이미지 크기 명시
    confidence_threshold=0.8,    # 신뢰도 임계값
    device='auto'                # 'auto', 'cpu', 'cuda', 'mps'
)

# 4. 배치 예측
results = model.predict_batch([
    'captcha1.png',
    'captcha2.png',
    'captcha3.png'
])

# 5. 바이트에서 예측
with open('captcha.png', 'rb') as f:
    result = model.predict_bytes(f.read())

# 6. numpy 배열에서 예측
import numpy as np
image_array = np.array(...)  # (200, 50) or (200, 50, 1)
result = model.predict_array(image_array)

# 7. 상세 결과 (신뢰도 포함)
result = model.predict('captcha.png', return_confidence=True)
print(result.text)        # '023062'
print(result.confidence)  # 0.95
```

#### 학습 API
```python
from captcha_cracker import CaptchaTrainer

# 학습 설정
trainer = CaptchaTrainer(
    image_size=(200, 50),
    model_config={
        'conv_filters': [32, 64],
        'lstm_units': [128, 64],
        'dropout_rate': 0.2
    }
)

# 데이터 로드 및 학습
trainer.load_data(['data/train_numbers_only/', 'data/train_numbers_only_2/'])
trainer.train(
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=['tensorboard', 'checkpoint']
)

# 모델 저장
trainer.save('model/weights_v2.h5')
```

### API 비교표

| 기능 | 구 API (v0.0.7) | 신 API (v1.0.0) | 개선점 |
|------|----------------|----------------|--------|
| 클래스명 | `ApplyModel` | `CaptchaModel` | 더 직관적 |
| 로드 방식 | `__init__()` | `.load()` 클래스 메서드 | 팩토리 패턴 |
| 컨텍스트 매니저 | ❌ | ✅ | 리소스 자동 관리 |
| 배치 예측 | ❌ | ✅ | 성능 향상 |
| 신뢰도 반환 | ❌ | ✅ | 결과 검증 가능 |
| 타입 힌트 | ❌ | ✅ | IDE 지원 |
| Docstring | ❌ | ✅ | 문서화 |
| 에러 핸들링 | ⚠️ | ✅ | 명확한 예외 |

## 마이그레이션 계획

### Phase 1: 환경 및 기반 업데이트 (Week 1)

#### 1.1 개발 환경 구성
- [ ] Python 3.12 설치 및 가상환경 생성
  ```bash
  pyenv install 3.12.7
  pyenv local 3.12.7
  python -m venv .venv
  source .venv/bin/activate
  ```

- [ ] 의존성 업데이트
  ```bash
  pip install --upgrade pip setuptools wheel
  pip install tensorflow==2.18.0
  pip install pillow==10.4.0
  pip install numpy==1.26.4
  ```

#### 1.2 Deprecated API 교체
- [ ] `layers.experimental.preprocessing.StringLookup` → `layers.StringLookup`
- [ ] `tf.data.experimental.AUTOTUNE` → `tf.data.AUTOTUNE`
- [ ] `keras.backend.ctc_batch_cost` → `tf.nn.ctc_loss`
- [ ] `keras.backend.ctc_decode` → `tf.nn.ctc_greedy_decoder`

#### 1.3 기본 동작 검증
- [ ] 기존 모델 가중치 로드 테스트
- [ ] 샘플 이미지로 예측 테스트
- [ ] 정확도 벤치마크 측정 (회귀 방지)

### Phase 2: 코드 구조 개선 (Week 2)

#### 2.1 패키지 구조 재설계
```
captcha_cracker/                    # 패키지명 변경 (snake_case)
├── __init__.py                     # 공개 API export
├── model.py                        # CaptchaModel 클래스
├── trainer.py                      # CaptchaTrainer 클래스
├── layers.py                       # CTCLayer 및 커스텀 레이어
├── preprocessing.py                # 이미지 전처리 유틸리티
├── utils.py                        # 헬퍼 함수
├── exceptions.py                   # 커스텀 예외
└── config.py                       # 설정 관리
```

#### 2.2 코드 중복 제거
- [ ] `build_model()` 메서드를 독립 함수로 추출 → `model.py`
- [ ] `CreateModel`과 `ApplyModel`을 `CaptchaTrainer`와 `CaptchaModel`로 재작성
- [ ] 공통 로직을 `utils.py`로 분리

#### 2.3 Dead Code 제거
- [ ] `ApplyModel.split_data()` 메서드 삭제
- [ ] 사용하지 않는 import 정리

### Phase 3: 코드 품질 향상 (Week 2-3)

#### 3.1 타입 힌트 추가
```python
from typing import Union, Tuple, List, Optional
from pathlib import Path
import numpy as np

def predict(
    self,
    image: Union[str, Path, bytes, np.ndarray],
    return_confidence: bool = False
) -> Union[str, PredictionResult]:
    ...
```

#### 3.2 Docstring 작성 (Google Style)
```python
class CaptchaModel:
    """캡챠 이미지 인식 모델.

    사전 학습된 가중치를 로드하여 캡챠 이미지에서 텍스트를 추출합니다.
    CTC (Connectionist Temporal Classification) 기반 OCR 모델을 사용합니다.

    Examples:
        >>> model = CaptchaModel.load('weights.h5')
        >>> result = model.predict('captcha.png')
        >>> print(result)
        '023062'

    Attributes:
        model: 로드된 Keras 모델
        image_size: 입력 이미지 크기 (width, height)
        characters: 인식 가능한 문자 목록
    """
```

#### 3.3 에러 핸들링 개선
```python
# exceptions.py
class CaptchaCrackerError(Exception):
    """Base exception for CaptchaCracker"""

class ModelLoadError(CaptchaCrackerError):
    """Failed to load model weights"""

class InvalidImageError(CaptchaCrackerError):
    """Invalid image format or size"""

class PredictionError(CaptchaCrackerError):
    """Failed to predict captcha"""
```

#### 3.4 로깅 시스템 추가
```python
import logging

logger = logging.getLogger('captcha_cracker')

# 사용자는 로깅 레벨 설정 가능
CaptchaModel.set_log_level('DEBUG')
```

### Phase 4: 테스트 및 검증 (Week 3)

#### 4.1 테스트 구조
```
tests/
├── __init__.py
├── conftest.py                     # pytest fixtures
├── test_model.py                   # CaptchaModel 테스트
├── test_trainer.py                 # CaptchaTrainer 테스트
├── test_preprocessing.py           # 전처리 테스트
├── test_integration.py             # 통합 테스트
└── fixtures/
    ├── sample_captcha.png
    └── weights_test.h5
```

#### 4.2 테스트 케이스
- [ ] 단위 테스트 작성 (pytest)
  - 모델 로드 테스트
  - 예측 함수 테스트
  - 전처리 함수 테스트
  - 에러 케이스 테스트

- [ ] 통합 테스트
  - End-to-end 예측 플로우
  - 배치 예측 테스트
  - 다양한 입력 형식 테스트

- [ ] 성능 테스트
  - 예측 속도 벤치마크
  - 메모리 사용량 측정

#### 4.3 커버리지 목표
```bash
pytest --cov=captcha_cracker --cov-report=html
# 목표: 80% 이상
```

### Phase 5: 문서화 (Week 3-4)

#### 5.1 README 업데이트
- [ ] 새로운 API 사용법
- [ ] 설치 가이드 (Python 3.12+)
- [ ] 예제 코드
- [ ] 마이그레이션 가이드 (v0.x → v1.0)

#### 5.2 API 문서 생성 (Sphinx)
```bash
docs/
├── conf.py
├── index.rst
├── api/
│   ├── model.rst
│   ├── trainer.rst
│   └── utils.rst
└── guides/
    ├── quickstart.rst
    ├── migration.rst
    └── advanced.rst
```

#### 5.3 CHANGELOG 작성
```markdown
# Changelog

## [1.0.0] - 2025-XX-XX

### Breaking Changes
- Python 3.12+ 필수
- API 완전 재설계 (하위 호환성 없음)
- 패키지명 변경: `CaptchaCracker` → `captcha_cracker`

### Added
- 새로운 `CaptchaModel` API
- 배치 예측 지원
- 신뢰도 점수 반환
- 타입 힌트 전면 적용
- 컨텍스트 매니저 지원

### Changed
- TensorFlow 2.5 → 2.18
- Deprecated API 교체
- 코드 구조 개선

### Fixed
- 보안 취약점 (Pillow 업그레이드)
- 메모리 누수 수정
```

### Phase 6: 배포 준비 (Week 4)

#### 6.1 패키지 설정 업데이트
```python
# setup.py 또는 pyproject.toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "captcha-cracker"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "tensorflow>=2.18.0,<2.19.0",
    "pillow>=10.4.0",
    "numpy>=1.26.0,<2.0.0",
]
```

#### 6.2 CI/CD 파이프라인 구성
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    - name: Run tests
      run: |
        pytest --cov=captcha_cracker
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

#### 6.3 배포
- [ ] GitHub 릴리스 생성 (v1.0.0)
- [ ] PyPI 배포
  ```bash
  python -m build
  twine upload dist/*
  ```
- [ ] 문서 사이트 배포 (GitHub Pages)

## 의존성 관리

### requirements.txt
```txt
# Production
tensorflow>=2.18.0,<2.19.0
pillow>=10.4.0
numpy>=1.26.0,<2.0.0
```

### requirements-dev.txt
```txt
# Development
pytest>=8.0.0
pytest-cov>=4.1.0
pytest-benchmark>=4.0.0

# Code quality
black>=24.0.0
isort>=5.13.0
flake8>=7.0.0
mypy>=1.11.0
pylint>=3.0.0

# Documentation
sphinx>=7.0.0
sphinx-rtd-theme>=2.0.0
sphinx-autodoc-typehints>=2.0.0

# Build
build>=1.0.0
twine>=5.0.0
```

### pyproject.toml (권장)
```toml
[project]
name = "captcha-cracker"
version = "1.0.0"
description = "Modern OCR library for captcha recognition using deep learning"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "tensorflow>=2.18.0,<2.19.0",
    "pillow>=10.4.0",
    "numpy>=1.26.0,<2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "black>=24.0.0",
    "isort>=5.13.0",
    "mypy>=1.11.0",
]

[tool.black]
line-length = 100
target-version = ['py312']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --cov=captcha_cracker --cov-report=term-missing"
```

## 가중치 파일 처리 (결정 필요)

### 배경 설명

프로젝트에는 두 개의 가중치 파일이 있습니다:
- `model/weights.h5` - 원본 학습된 모델 가중치
- `model/weights_v2.h5` - 업데이트된 모델 가중치

**가중치 파일이란?**
딥러닝 모델이 학습 과정에서 최적화한 파라미터(가중치)를 저장한 파일입니다. 이 파일이 있으면 모델을 처음부터 다시 학습하지 않고도 바로 예측에 사용할 수 있습니다.

### 옵션 분석

#### 옵션 A: 가중치 보존 (권장)
**장점:**
- 기존 학습 결과 유지
- 즉시 사용 가능
- 재학습 비용 없음

**단점:**
- TensorFlow 버전 호환성 검증 필요
- 일부 레이어 로드 시 경고 발생 가능

**작업:**
1. TensorFlow 2.18에서 기존 `.h5` 파일 로드 테스트
2. 호환성 문제 발견 시 가중치 변환
3. 새로운 형식으로 재저장 (`.keras` 또는 SavedModel)

#### 옵션 B: 재학습
**장점:**
- 최신 TensorFlow API로 깔끔하게 학습
- 하이퍼파라미터 튜닝 기회
- 더 나은 성능 가능

**단점:**
- 재학습 시간 필요 (GPU 기준 수 시간)
- 기존 성능 보장 불가

**작업:**
1. `data.zip` 압축 해제
2. 새로운 학습 스크립트 작성
3. 학습 실행 (100 epochs)
4. 성능 비교

#### 옵션 C: 병행 (최소 위험)
**방식:**
1. 먼저 기존 가중치로 마이그레이션 (옵션 A)
2. 백그라운드에서 재학습 수행 (옵션 B)
3. 성능 비교 후 더 나은 것 선택

### 권장 사항
**🔹 옵션 A (가중치 보존)를 먼저 시도하고, 문제 발생 시 옵션 B로 전환**

이유:
- 리팩토링 범위는 "레거시 → 최신 버전 마이그레이션"
- 모델 개선은 별도 문서(FUTURE_IMPROVEMENTS.md)에서 다룸
- 위험 최소화

## 예상 일정

| Phase | 작업 내용 | 기간 | 마일스톤 |
|-------|----------|------|----------|
| Phase 1 | 환경 구성 및 API 교체 | 1주 | ✅ TensorFlow 2.18 동작 |
| Phase 2 | 코드 구조 개선 | 1주 | ✅ 새로운 패키지 구조 |
| Phase 3 | 코드 품질 향상 | 1-2주 | ✅ 타입 힌트, 문서화 |
| Phase 4 | 테스트 작성 | 1주 | ✅ 80%+ 커버리지 |
| Phase 5 | 문서화 | 1주 | ✅ Sphinx 문서 |
| Phase 6 | 배포 준비 | 1주 | ✅ v1.0.0 릴리스 |

**총 예상 기간**: 6-7주

## 위험 요소 및 대응책

### 위험 1: TensorFlow 호환성 문제
**가능성**: 중간
**영향**: 높음

**완화책:**
- 단계별 검증 (각 API 교체 후 테스트)
- 기존 가중치로 예측 테스트 필수
- 문제 발생 시 중간 버전 (TensorFlow 2.15) 거쳐 단계적 업그레이드

### 위험 2: API 변경으로 인한 사용자 혼란
**가능성**: 높음
**영향**: 중간

**완화책:**
- 상세한 마이그레이션 가이드 작성
- v0.x와 v1.0 비교표 제공
- Major 버전 업그레이드로 명확히 표시 (Breaking Change)
- 예제 코드 다수 제공

### 위험 3: 성능 저하
**가능성**: 낮음
**영향**: 높음

**완화책:**
- 리팩토링 전 성능 벤치마크 측정
- 각 단계에서 성능 비교
- 회귀 발견 시 프로파일링 및 최적화

### 위험 4: 일정 지연
**가능성**: 중간
**영향**: 낮음

**완화책:**
- 버퍼 기간 포함한 일정 수립
- Phase 단위로 점진적 배포 가능
- MVP 우선 (Phase 1-3), 나머지는 후속 배포

## 성공 기준

### 필수 (Must Have)
- [ ] Python 3.12 지원
- [ ] TensorFlow 2.18 동작
- [ ] Deprecated API 제거 (0개)
- [ ] 기존 가중치로 예측 성공
- [ ] 새로운 API 구현
- [ ] 기본 테스트 작성 (커버리지 60%+)

### 권장 (Should Have)
- [ ] 타입 힌트 적용 (80%+)
- [ ] Docstring 작성 (public API 100%)
- [ ] 테스트 커버리지 80%+
- [ ] CI/CD 파이프라인 구축
- [ ] 문서 사이트 배포

### 선택 (Nice to Have)
- [ ] 배치 예측 구현
- [ ] 신뢰도 점수 반환
- [ ] 로깅 시스템
- [ ] Docker 이미지
- [ ] 성능 벤치마크

## 체크리스트

### 사전 준비
- [ ] 현재 코드 백업 (Git tag: v0.0.7)
- [ ] 성능 벤치마크 측정 (baseline)
- [ ] 테스트 이미지 준비

### Phase 1
- [ ] Python 3.12 환경 구성
- [ ] TensorFlow 2.18 설치
- [ ] Deprecated API 교체
- [ ] 기본 동작 검증

### Phase 2
- [ ] 새로운 패키지 구조 생성
- [ ] 코드 중복 제거
- [ ] Dead code 제거

### Phase 3
- [ ] 타입 힌트 추가
- [ ] Docstring 작성
- [ ] 에러 핸들링 개선
- [ ] 로깅 추가

### Phase 4
- [ ] pytest 설정
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 커버리지 80%+ 달성

### Phase 5
- [ ] README 업데이트
- [ ] 마이그레이션 가이드 작성
- [ ] Sphinx 문서 생성
- [ ] CHANGELOG 작성

### Phase 6
- [ ] `pyproject.toml` 설정
- [ ] CI/CD 파이프라인 구축
- [ ] PyPI 배포
- [ ] GitHub 릴리스 (v1.0.0)

## 참고 자료

### 공식 문서
- [TensorFlow 2.18 Release Notes](https://github.com/tensorflow/tensorflow/releases/tag/v2.18.0)
- [TensorFlow Migration Guide](https://www.tensorflow.org/guide/migrate)
- [Python 3.12 Release Notes](https://docs.python.org/3.12/whatsnew/3.12.html)
- [Keras 3 API Documentation](https://keras.io/api/)

### 도구 문서
- [pytest Documentation](https://docs.pytest.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Black Code Formatter](https://black.readthedocs.io/)

### 참고 프로젝트
- TensorFlow 마이그레이션 예제
- 모던 Python 패키지 구조

## 다음 단계

1. ✅ **가중치 파일 처리 방식 결정**
   - 옵션 A (보존), B (재학습), C (병행) 중 선택

2. **Phase 1 시작**
   - Python 3.12 환경 구성
   - TensorFlow 2.18 설치
   - Deprecated API 교체 시작

3. **정기 체크포인트**
   - 매 Phase 종료 시 코드 리뷰
   - 성능 벤치마크 비교
   - 문서 업데이트

---

**문서 버전**: 1.0
**최종 수정**: 2025-11-19
**상태**: 🟡 가중치 처리 방식 결정 대기 중
