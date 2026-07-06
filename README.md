# compliance_agent

금융 도메인 개인정보 마스킹 데이터를 생성하고, 생성된 데이터를 바탕으로 토큰 분류 모델 실험을 진행하는 프로젝트입니다.

## 구조

- `v1_text/text_generator_v1.py`: OpenRouter 기반 마스킹 데이터 생성 스크립트
- `v1_text/llm/openrouter.py`: OpenRouter API 호출 래퍼
- `v1_text/llm/prompt/`: 데이터셋 유형별 시스템 프롬프트
- `v1_text/data/`: 생성된 CSV 데이터셋
- `v1_text/train_model.ipynb`: 토큰 분류 모델 학습 실험 노트북
- `v1_text/training/pipeline.py`: 재사용 가능한 학습, 평가, 추론 파이프라인 모듈
- `v1_text/train_model_v1.py`: 학습 파이프라인 CLI 진입점
- `docs/specs/financial_text_span_schema.md`: canonical span 스키마 문서 명세
- `docs/specs/financial_text_span_schema.json`: canonical span JSON Schema

## 요구 사항

- Python 3.10 이상 권장
- OpenRouter API 키

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 환경 변수 설정

`.env_example`을 복사해 `.env`를 만들고 값을 채웁니다.

```env
OPENROUTER_API_KEY=your-openrouter-api-key
MODEL=openai/gpt-4o-mini
TEMPERATURE=0.2
MAX_TOKENS=1024
SYSTEM_PROMPT=prompt_counsel_v1.md
DATASET_NAME=고객상담_마스킹_v1
TOPIC=금융 고객 상담에서 개인정보를 마스킹한 텍스트
DATA_NUM=3
```

`SYSTEM_PROMPT`는 `v1_text/llm/prompt/` 아래 파일명 중 하나여야 합니다.

## 데이터 생성 실행

저장소 루트에서 아래 명령으로 실행합니다.

```powershell
python -m v1_text.text_generator_v1
```

성공하면 `v1_text/data/` 아래에 CSV가 생성됩니다.

## 학습 실험

`v1_text/train_model.ipynb`는 기존 실험 노트북으로 유지하고, 실제 재사용 경로는 Python 모듈로 분리했습니다.

학습 파이프라인은 아래 명령으로 실행할 수 있습니다.

```powershell
python -m v1_text.train_model_v1
```

실제 CSV를 사용하려면 경로를 넘기면 됩니다.

```powershell
python -m v1_text.train_model_v1 --train-csv v1_text/data/customer_service_conversation.csv --output-dir artifacts/bert_mask_model
```

기본값으로는 노트북 예시 데이터를 사용합니다.

## 스키마 명세

텍스트 마스킹 데이터의 canonical span schema는 아래 두 파일에 정의돼 있습니다.

- `docs/specs/financial_text_span_schema.md`
- `docs/specs/financial_text_span_schema.json`

향후 프롬프트 출력, CSV 로딩, 평가 입력 형식은 이 스키마를 기준으로 통일하는 것이 원칙입니다.

## 현재 한계

- 자동화 테스트가 아직 없습니다.
- 평가 스크립트와 실데이터 기준 학습 검증은 아직 별도 태스크로 남아 있습니다.
- 엔티티 품질 검증 규칙이 코드로 자동화되어 있지 않습니다.
