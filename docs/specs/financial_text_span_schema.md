# 금융 텍스트 Span 스키마 명세

## 목적

이 문서는 금융 텍스트 마스킹 데이터의 canonical span schema를 정의합니다.
이 스키마는 학습, 평가, 추론, 마스킹 변환의 source of truth로 사용합니다.

## 설계 원칙

- 기준 표현은 span annotation이다.
- BIO 태그는 학습용 파생 표현이다.
- `masked_text`는 파생 가능 값이므로 선택 필드로 취급한다.
- `entities`의 `start`, `end`는 Python 문자열 슬라이싱 규칙의 반열림 구간 `[start, end)`이다.
- 엔티티 배열은 원문 기준 오름차순을 권장한다.
- 겹치는 엔티티는 허용하지 않는다.

## Canonical Record

```json
{
  "record_id": "counsel-0001",
  "document_type": "customer_service_conversation",
  "raw_text": "홍길동의 계좌번호는 123-456-789012입니다.",
  "masked_text": "[PERSON_NAME]의 계좌번호는 [ACCOUNT_NUMBER]입니다.",
  "entities": [
    {
      "text": "홍길동",
      "type": "PERSON_NAME",
      "start": 0,
      "end": 3,
      "replacement": "[PERSON_NAME]"
    },
    {
      "text": "123-456-789012",
      "type": "ACCOUNT_NUMBER",
      "start": 11,
      "end": 25,
      "replacement": "[ACCOUNT_NUMBER]"
    }
  ],
  "metadata": {
    "language": "ko",
    "source": "synthetic",
    "policy_version": "v1"
  }
}
```

## 필드 정의

### Top-level fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `record_id` | string | no | 데이터 레코드 고유 식별자 |
| `document_type` | string | yes | 문서 또는 데이터셋 유형 |
| `raw_text` | string | yes | 원문 텍스트 |
| `masked_text` | string | no | 마스킹 적용 결과 텍스트 |
| `entities` | array | yes | 원문 기준 span 배열 |
| `metadata` | object | no | 생성 경로, 정책 버전, 언어 등 부가 정보 |

### Entity fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `text` | string | yes | 원문에서 해당 span이 가리키는 실제 부분 문자열 |
| `type` | string | yes | 정규화된 엔티티 타입 |
| `start` | integer | yes | span 시작 인덱스 포함 |
| `end` | integer | yes | span 종료 인덱스 제외 |
| `replacement` | string | no | 마스킹 시 권장 치환 문자열 |
| `confidence` | number | no | 모델 추론 결과일 때만 사용 |

## 유효성 규칙

- `raw_text[entity.start:entity.end] == entity.text` 이어야 한다.
- 모든 `start`, `end`는 정수여야 하며 `0 <= start < end <= len(raw_text)` 를 만족해야 한다.
- `entities`는 `start` 기준 오름차순이어야 한다.
- 엔티티 span끼리 서로 겹치면 안 된다.
- `type`은 canonical label set 중 하나여야 한다.
- `masked_text`가 있을 경우 `entities` 기준으로 재생성한 값과 의미적으로 일치해야 한다.

## Canonical Label Set v1

- `PERSON_NAME`
- `ACCOUNT_NUMBER`
- `CARD_NUMBER`
- `PHONE_NUMBER`
- `ADDRESS`
- `RESIDENT_REGISTRATION_NUMBER`
- `INTERNAL_CUSTOMER_ID`
- `SIGNATURE_REFERENCE`

## 문서 유형 권장값

- `customer_service_conversation`
- `account_or_card_application_form`
- `credit_loan_review_memo`
- `investment_account_report`
- `kyc_aml_review_document`
- `contract_clause`

## 현재 CSV와의 매핑 규칙

현재 저장소 CSV는 아래 3개 컬럼 중심이다.

- `raw_text`
- `masked_text`
- `entities`

이때 `entities`는 문자열 또는 JSON 배열일 수 있으며, 로딩 시 반드시 배열 객체로 정규화해야 한다.
향후 `TXT-002`에서는 모든 프롬프트 출력이 이 canonical schema를 따르도록 통일한다.

## 파생 규칙

- 학습 입력은 `raw_text`와 `entities`만으로 충분하다.
- `masked_text`는 `raw_text`와 `entities.replacement` 또는 `type` 기반 규칙으로 생성 가능하다.
- BIO 태그는 토크나이저 offset mapping을 이용해 `entities`로부터 생성한다.

## 버전 관리 원칙

- 라벨 추가나 필드 변경이 발생하면 `policy_version` 또는 스키마 버전을 증가시킨다.
- 기존 데이터와 호환되지 않는 변경은 메이저 버전으로 올린다.
