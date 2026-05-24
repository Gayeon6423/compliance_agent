금융 고객 상담 문장을 대상으로 고품질 PII 마스킹 데이터셋을 생성하세요.

반환 형식 규칙(필수):
1) 반드시 JSON 배열(Array)만 반환하세요. 마크다운 코드펜스, 설명 문장, 주석 금지.
2) 배열의 각 원소는 아래 키를 반드시 포함해야 합니다.
- raw_text: 원문 문장 (string)
- entities: 민감정보 엔티티 목록 (array)
- masked_text: 엔티티 타입 토큰으로 치환된 문장 (string)
3) entities의 각 원소는 아래 키를 반드시 포함해야 합니다.
- text: raw_text에 실제로 존재하는 원문 span (string)
- type: 엔티티 타입 (string, 예: PERSON_NAME, ACCOUNT_NUMBER, CARD_NUMBER, PHONE_NUMBER, RESIDENT_ID, ADDRESS, TRANSACTION_AMOUNT)
- start: raw_text에서 text가 시작하는 문자 인덱스 (0-based, inclusive)
- end: raw_text에서 text가 끝나는 문자 인덱스 (0-based, exclusive)

정합성 규칙(필수):
- raw_text[start:end] == text 를 반드시 만족해야 합니다.
- 같은 문장 내 entities는 start 오름차순으로 정렬하세요.
- masked_text는 raw_text의 각 엔티티 text를 정확히 [TYPE]으로 치환한 결과여야 합니다.
- end는 포함이 아니라 제외(exclusive)입니다.
- 숫자 금액은 통화 단위(예: 원)는 엔티티 밖에 두고, 숫자 부분만 엔티티로 태깅하세요.
- 모든 값은 JSON 타입을 지켜 출력하세요(start/end는 정수).

출력 예시 형식(스키마 참고용):
[
	{
		"raw_text": "홍길동 고객님 계좌 123-456-789012에서 3000000원이 출금되었습니다.",
		"entities": [
			{"text": "홍길동", "type": "PERSON_NAME", "start": 0, "end": 3},
			{"text": "123-456-789012", "type": "ACCOUNT_NUMBER", "start": 10, "end": 24},
			{"text": "3000000", "type": "TRANSACTION_AMOUNT", "start": 27, "end": 34}
		],
		"masked_text": "[PERSON_NAME] 고객님 계좌 [ACCOUNT_NUMBER]에서 [TRANSACTION_AMOUNT]원이 출금되었습니다."
	}
]
 