계좌 개설, 카드 발급, 고객 등록, 결제계좌 등록 등에 사용되는 신청서와 신청서형 서식을 대상으로 고품질 금융 마스킹 데이터셋을 생성하세요.

반환 형식 규칙(필수):
1) 반드시 JSON 배열(Array)만 반환하세요. 마크다운 코드펜스, 설명 문장, 주석 금지.
2) 배열의 각 원소는 아래 키를 반드시 포함해야 합니다.
- raw_text: 원문 신청서, 서식, 접수 메모 (string)
- entities: 민감정보 엔티티 목록 (array)
- masked_text: 엔티티 타입 토큰으로 치환된 신청서, 서식 (string)
3) entities의 각 원소는 아래 키를 반드시 포함해야 합니다.
- text: raw_text에 실제로 존재하는 원문 span (string)
- type: 엔티티 타입 (string, 예: PERSON_NAME, ENGLISH_NAME, PHONE_NUMBER, ADDRESS, WORKPLACE_NAME, WORKPLACE_ADDRESS, PAYMENT_ACCOUNT_NUMBER, ID_ISSUANCE_DATE, ID_NUMBER, EMAIL)
- start: raw_text에서 text가 시작하는 문자 인덱스 (0-based, inclusive)
- end: raw_text에서 text가 끝나는 문자 인덱스 (0-based, exclusive)

정합성 규칙(필수):
- raw_text[start:end] == text 를 반드시 만족해야 합니다.
- 같은 문단 내 entities는 start 오름차순으로 정렬하세요.
- masked_text는 raw_text의 각 엔티티 text를 정확히 [TYPE]으로 치환한 결과여야 합니다.
- end는 포함이 아니라 제외(exclusive)입니다.
- 성명, 연락처, 주소, 직장 정보, 결제계좌, 이메일, 신분증 정보는 모두 민감정보로 태깅할 수 있습니다.
- 모든 값은 JSON 타입을 지켜 출력하세요(start/end는 정수).

생성 가이드:
- 계좌 개설, 체크카드/신용카드 발급, 고객정보 등록, 자동이체 등록, 전자서명 동의서 같은 문서를 자연스럽게 작성하세요.
- 신청인 정보, 직장 정보, 결제 정보, 본인확인, 동의 항목이 섞일 수 있습니다.
- 신청서 느낌이 나도록 항목별 레이아웃이나 라벨을 포함해도 됩니다.
- 출력 개수는 사용자 요청 수량에 맞추세요.

출력 예시 형식(스키마 참고용):
[
	{
		"raw_text": "신용카드 발급 신청서입니다. 성명 한지우, 휴대전화 010-4821-7732, 주소 서울특별시 송파구 올림픽로 212, 102동 1804호, 결제계좌번호 088-123-456789입니다.",
		"entities": [
			{"text": "한지우", "type": "PERSON_NAME", "start": 17, "end": 20},
			{"text": "010-4821-7732", "type": "PHONE_NUMBER", "start": 27, "end": 40},
			{"text": "서울특별시 송파구 올림픽로 212, 102동 1804호", "type": "ADDRESS", "start": 43, "end": 71},
			{"text": "088-123-456789", "type": "PAYMENT_ACCOUNT_NUMBER", "start": 80, "end": 94}
		],
		"masked_text": "신용카드 발급 신청서입니다. 성명 [PERSON_NAME], 휴대전화 [PHONE_NUMBER], 주소 [ADDRESS], 결제계좌번호 [PAYMENT_ACCOUNT_NUMBER]입니다."
	}
]
