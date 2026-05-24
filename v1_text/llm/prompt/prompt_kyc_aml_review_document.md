고객확인(KYC), 강화된 고객확인(EDD), 자금출처 확인, 이상거래 모니터링, 의심거래 검토 메모를 대상으로 고품질 금융 마스킹 데이터셋을 생성하세요.

반환 형식 규칙(필수):
1) 반드시 JSON 배열(Array)만 반환하세요. 마크다운 코드펜스, 설명 문장, 주석 금지.
2) 배열의 각 원소는 아래 키를 반드시 포함해야 합니다.
- raw_text: 원문 문서, 메모, 검토 기록 (string)
- entities: 민감정보 엔티티 목록 (array)
- masked_text: 엔티티 타입 토큰으로 치환된 문서, 메모 (string)
3) entities의 각 원소는 아래 키를 반드시 포함해야 합니다.
- text: raw_text에 실제로 존재하는 원문 span (string)
- type: 엔티티 타입 (string, 예: CUSTOMER_NAME, CUSTOMER_ID, ACCOUNT_NUMBER, SOURCE_OF_FUNDS, SUSPECTED_TRANSACTION, BENEFICIARY_NAME, ADDRESS, RISK_LEVEL, DOCUMENT_NUMBER)
- start: raw_text에서 text가 시작하는 문자 인덱스 (0-based, inclusive)
- end: raw_text에서 text가 끝나는 문자 인덱스 (0-based, exclusive)

정합성 규칙(필수):
- raw_text[start:end] == text 를 반드시 만족해야 합니다.
- 같은 문단 내 entities는 start 오름차순으로 정렬하세요.
- masked_text는 raw_text의 각 엔티티 text를 정확히 [TYPE]으로 치환한 결과여야 합니다.
- end는 포함이 아니라 제외(exclusive)입니다.
- 자금출처, 의심거래 내용, 수취인, 고객명, 고객번호, 계좌번호, 주소, 위험등급, 문서번호는 모두 민감정보로 태깅할 수 있습니다.
- 모든 값은 JSON 타입을 지켜 출력하세요(start/end는 정수).

생성 가이드:
- 실제 AML/KYC 내부 검토 문서처럼 자연스럽게 작성하세요.
- 고객 기본정보, 계좌 및 거래 정보, 고객 소명, 검토 의견, 후속 조치가 섞일 수 있습니다.
- 고액 현금 입금, 해외송금, 반복 거래, 의심 사유, 증빙자료 요청 같은 표현을 포함하세요.
- 출력 개수는 사용자 요청 수량에 맞추세요.

출력 예시 형식(스키마 참고용):
[
	{
		"raw_text": "AML 의심거래 검토 메모입니다. 고객명 노태준, 고객번호 CUST-AML-778201, 주계좌번호 088-777-123456에 대해 단기간 반복 고액 현금 입금 후 해외송금 패턴이 확인되었습니다.",
		"entities": [
			{"text": "노태준", "type": "CUSTOMER_NAME", "start": 18, "end": 21},
			{"text": "CUST-AML-778201", "type": "CUSTOMER_ID", "start": 28, "end": 44},
			{"text": "088-777-123456", "type": "ACCOUNT_NUMBER", "start": 53, "end": 67},
			{"text": "단기간 반복 고액 현금 입금 후 해외송금", "type": "SUSPECTED_TRANSACTION", "start": 74, "end": 94}
		],
		"masked_text": "AML 의심거래 검토 메모입니다. 고객명 [CUSTOMER_NAME], 고객번호 [CUSTOMER_ID], 주계좌번호 [ACCOUNT_NUMBER]에 대해 [SUSPECTED_TRANSACTION] 패턴이 확인되었습니다."
	}
]
