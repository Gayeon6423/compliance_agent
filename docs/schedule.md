# Schedule Log

이 문서는 작업 이력의 source of truth입니다. 새로운 작업을 시작하거나 상태가 바뀔 때마다 아래 표에 한 줄씩 추가합니다. 최신 기록을 위에 추가하세요.

## Status Vocabulary

- `planned`
- `started`
- `in-progress`
- `blocked`
- `completed`

## WBS

### 1단계 텍스트 마스킹 모델

- TXT-001: 금융 텍스트 span 엔티티 스키마 확정
- TXT-002: 프롬프트별 출력 포맷을 하나의 span schema로 통일
- TXT-003: train_model 노트북을 재사용 가능한 Python 학습 모듈로 분리
- TXT-004: 모델 출력 span을 마스킹 텍스트로 바꾸는 변환기 구현
- TXT-005: 엔티티 기준 precision, recall, F1, 문서 단위 성공률 측정 스크립트 작성

### 2단계 모델 비교 및 고도화

- TXT-006: BERT, RoBERTa, Small LM, Hybrid 비교 실험 표준화

### 3단계 이미지 계약서 마스킹

- IMG-001: OCR -> NER -> bbox 매핑 방식의 이미지 마스킹 설계 문서 작성
- IMG-002: Detection/LayoutLM/VLM 기반 대안 아키텍처 비교

## Work Log

| Date | Task ID | Task | Status | Note | Artifact |
| --- | --- | --- | --- | --- | --- |
| 2026-07-06 | OPS-002 | dashboard UI 단순화 | completed | 미리보기 제거, 액션 버튼을 예정/진행/완료로 변경, 최근 활동을 아래 스케줄 표로 이동 | docs/dashboard.html, docs/workflow.md |
| 2026-07-06 | OPS-002 | 문서 구축 항목 정리 | completed | dashboard 미리보기를 제거하고 workflow의 OPS 항목을 실행 환경과 문서 구축 두 줄로 단순화 | docs/workflow.md, docs/dashboard.html |
| 2026-07-06 | OPS-006 | workflow.md 쉬운 한국어 표현으로 정리 | completed | workflow 섹션 제목과 작업 보드 표현을 더 쉬운 한국어로 바꾸고 dashboard 저장 결과와 맞춤 | docs/workflow.md, docs/dashboard.html |
| 2026-07-06 | OPS-005 | dashboard file:// 로컬 열기 지원 | completed | fetch 실패 시 내장 markdown fallback과 수동 파일 불러오기 버튼을 추가 | docs/dashboard.html |
| 2026-07-06 | OPS-004 | dashboard markdown 직접 저장 기능 추가 | completed | File System Access API 기반 저장과 다운로드 fallback을 dashboard에 추가 | docs/dashboard.html |
| 2026-07-06 | TXT-003 | train_model 노트북 Python 모듈화 | completed | training 패키지와 CLI 진입점을 추가해 노트북 로직을 Python 모듈로 분리 | v1_text/training/pipeline.py, v1_text/train_model_v1.py |
| 2026-07-06 | TXT-001 | 금융 텍스트 span 엔티티 스키마 확정 | completed | canonical span 스키마의 문서형 명세와 JSON Schema 파일을 추가 | docs/specs/financial_text_span_schema.md, docs/specs/financial_text_span_schema.json |
| 2026-07-06 | OPS-004 | dashboard markdown 직접 저장 기능 추가 | in-progress | dashboard에서 workflow.md 와 schedule.md를 직접 저장할 수 있도록 기능 확장 시작 | docs/dashboard.html |
| 2026-07-06 | TXT-003 | train_model 노트북 Python 모듈화 | in-progress | 학습 노트북 로직을 재사용 가능한 학습 파이프라인 모듈로 분리 시작 | v1_text/train_model.ipynb |
| 2026-07-06 | TXT-001 | 금융 텍스트 span 엔티티 스키마 확정 | in-progress | span 스키마 명세 파일 설계 및 생성 시작 | docs/workflow.md |
| 2026-07-06 | OPS-003 | AGENTS.md 한국어화 | completed | AGENTS 운영 지침을 전체 한국어로 번역하고 표현을 정리 | AGENTS.md |
| 2026-07-06 | OPS-002 | Agent 운영 문서와 대시보드 구성 | completed | AGENTS, workflow, dashboard, schedule 포맷을 구축 | AGENTS.md, docs/workflow.md, docs/dashboard.html |
| 2026-07-06 | OPS-001 | 텍스트 생성기 실행성 개선 | completed | 경로 계산, .env 예시, README, requirements 정리 | v1_text/text_generator_v1.py, v1_text/llm/openrouter.py, README.md |
