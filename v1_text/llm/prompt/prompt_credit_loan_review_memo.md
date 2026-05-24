대출 심사, 신용평가, 승인/거절 판단, 내부 리스크 의견을 포함한 신용대출 심사 메모를 대상으로 고품질 금융 마스킹 데이터셋을 생성하세요.

반환 형식 규칙(필수):
1) 반드시 JSON 배열(Array)만 반환하세요. 마크다운 코드펜스, 설명 문장, 주석 금지.
2) 배열의 각 원소는 아래 키를 반드시 포함해야 합니다.
- raw_text: 원문 문서, 심사 메모, 의견서 (string)
- entities: 민감정보 엔티티 목록 (array)
- masked_text: 엔티티 타입 토큰으로 치환된 문서, 심사 메모 (string)
3) entities의 각 원소는 아래 키를 반드시 포함해야 합니다.
- text: raw_text에 실제로 존재하는 원문 span (string)
- type: 엔티티 타입 (string, 예: PERSON_NAME, CUSTOMER_ID, CREDIT_SCORE, DSR, DELINQUENCY_HISTORY, INCOME_AMOUNT, DEBT_AMOUNT, APPROVAL_LIMIT, REVIEW_OPINION, EMPLOYER_NAME)
- start: raw_text에서 text가 시작하는 문자 인덱스 (0-based, inclusive)
- end: raw_text에서 text가 끝나는 문자 인덱스 (0-based, exclusive)

정합성 규칙(필수):
- raw_text[start:end] == text 를 반드시 만족해야 합니다.
- 같은 문단 내 entities는 start 오름차순으로 정렬하세요.
- masked_text는 raw_text의 각 엔티티 text를 정확히 [TYPE]으로 치환한 결과여야 합니다.
- end는 포함이 아니라 제외(exclusive)입니다.
- 신용점수, DSR, 연체이력, 소득, 부채, 승인한도, 금리, 심사의견은 문서 유형상 민감정보로 태깅할 수 있습니다.
- 모든 값은 JSON 타입을 지켜 출력하세요(start/end는 정수).

생성 가이드:
- 실제 여신 심사 보고서, 내부 심사 메모, 승인 조건 메모처럼 자연스럽게 작성하세요.
- 신청 요약, 신용정보, 소득 및 상환능력, 심사 결과, 담당자 의견이 섞일 수 있습니다.
- 승인/조건부 승인/거절 사유를 적절히 포함하고, 정량 정보와 정성 의견이 함께 나오게 하세요.
- 출력 개수는 사용자 요청 수량에 맞추세요.

출력 예시 형식(스키마 참고용):
[
	{
		"raw_text": "개인신용대출 내부 심사 보고서입니다. 고객명 오준영, NICE 신용점수 842점, 추정 DSR 36.8%이며 승인한도는 38,000,000원으로 검토되었습니다.",
		"entities": [
			{"text": "오준영", "type": "PERSON_NAME", "start": 20, "end": 23},
			{"text": "842", "type": "CREDIT_SCORE", "start": 35, "end": 38},
			{"text": "36.8", "type": "DSR", "start": 51, "end": 55},
			{"text": "38,000,000", "type": "APPROVAL_LIMIT", "start": 66, "end": 76}
		],
		"masked_text": "개인신용대출 내부 심사 보고서입니다. 고객명 [PERSON_NAME], NICE 신용점수 [CREDIT_SCORE]점, 추정 DSR [DSR]%이며 승인한도는 [APPROVAL_LIMIT]원으로 검토되었습니다."
	}
]
