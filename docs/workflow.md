# 작업 흐름

이 문서는 프로젝트 구조, 현재 상태, 다음 작업을 관리하는 기준 문서입니다.

## 프로젝트 한눈에 보기

- 목표: 금융 문서와 텍스트에서 민감 정보를 안전하게 마스킹하는 파이프라인 구축
- 핵심 지표: 마스킹 누락 감소, 처리 시간 단축, 과다 마스킹 감소, 문서 유형 확장
- 진행 로드맵: 텍스트 마스킹 -> 모델 비교 및 고도화 -> 이미지 계약서 마스킹

## 구조 요약

### 텍스트 파이프라인

`text -> NER model -> span extraction -> masking transformation`

### 이미지 파이프라인

`image -> OCR -> text NER -> bbox mapping -> blur`

또는

`image -> document detection model -> bbox detection -> blur`

## 작업 보드

| ID | 단계 | 작업 영역 | 작업 내용 | 상태 | 우선순위 | 수정일 |
| --- | --- | --- | --- | --- | --- | --- |
| OPS-001 | 기반 | 실행 환경 | 텍스트 생성기 실행 경로와 환경 변수 정리 | 완료 | 높음 | 2026-07-06 |
| OPS-002 | 기반 | 문서 구축 | AGENTS.md, workflow.md, schedule.md, dashboard.html 구축 및 정리 | 완료 | 높음 | 2026-07-06 |
| TXT-001 | 1단계 | 데이터 스키마 | 금융 텍스트 span 엔티티 스키마 확정 | 완료 | 높음 | 2026-07-06 |
| TXT-002 | 1단계 | 데이터 생성 | 프롬프트별 출력 포맷을 하나의 span schema로 통일 | 예정 | 높음 | 2026-07-06 |
| TXT-003 | 1단계 | 학습 파이프라인 | train_model 노트북을 재사용 가능한 Python 학습 모듈로 분리 | 완료 | 높음 | 2026-07-06 |
| TXT-004 | 1단계 | 후처리 | 모델 출력 span을 마스킹 텍스트로 바꾸는 변환기 구현 | 예정 | 중간 | 2026-07-06 |
| TXT-005 | 1단계 | 평가 | 엔티티 기준 precision, recall, F1, 문서 단위 성공률 측정 스크립트 작성 | 예정 | 높음 | 2026-07-06 |
| TXT-006 | 2단계 | 모델 비교 | BERT, RoBERTa, Small LM, Hybrid 비교 실험 표준화 | 예정 | 중간 | 2026-07-06 |
| IMG-001 | 3단계 | OCR 파이프라인 | OCR -> NER -> bbox 매핑 방식의 이미지 마스킹 설계 문서 작성 | 예정 | 중간 | 2026-07-06 |
| IMG-002 | 3단계 | 비전 파이프라인 | Detection/LayoutLM/VLM 기반 대안 아키텍처 비교 | 예정 | 낮음 | 2026-07-06 |

## 완료된 작업

- 텍스트 생성기의 import 경로와 실행 경로를 안정화했다.
- 환경 변수 예시 파일과 실행 문서를 정리했다.
- 루트 README와 requirements 파일을 추가했다.
- 에이전트 운영용 문서 구조를 정리했다.
- 금융 텍스트 canonical span 스키마 명세 파일 추가.
- train_model 노트북 로직을 Python 학습 모듈과 CLI로 분리했다.
- dashboard에서 workflow.md 와 schedule.md 저장 기능을 추가했다.
- dashboard를 file:// 로 직접 열어도 사용할 수 있게 수정했다.
- workflow.md의 영어 중심 표현을 쉬운 한국어로 정리했다.
