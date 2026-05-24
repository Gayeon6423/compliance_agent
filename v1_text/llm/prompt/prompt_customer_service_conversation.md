고객센터 통화 STT, 채팅, 이메일, 민원 스레드 등 고객 응대 대화를 대상으로 고품질 금융 마스킹 데이터셋을 생성하세요.

반환 형식 규칙(필수):
1) 반드시 JSON 배열(Array)만 반환하세요. 마크다운 코드펜스, 설명 문장, 주석 금지.
2) 배열의 각 원소는 아래 키를 반드시 포함해야 합니다.
- raw_text: 원문 문장, 대화, 메모 (string)
- entities: 민감정보 엔티티 목록 (array)
- masked_text: 엔티티 타입 토큰으로 치환된 문장 또는 대화 (string)
3) entities의 각 원소는 아래 키를 반드시 포함해야 합니다.
- text: raw_text에 실제로 존재하는 원문 span (string)
- type: 엔티티 타입 (string, 예: PERSON_NAME, PHONE_NUMBER, CARD_NUMBER_LAST4, ACCOUNT_NUMBER_LAST4, APPROVAL_NUMBER, AUTH_CODE, TRANSACTION_AMOUNT, DATE_TIME, MERCHANT_NAME)
- start: raw_text에서 text가 시작하는 문자 인덱스 (0-based, inclusive)
- end: raw_text에서 text가 끝나는 문자 인덱스 (0-based, exclusive)

정합성 규칙(필수):
- raw_text[start:end] == text 를 반드시 만족해야 합니다.
- 같은 문장 또는 대화 내 entities는 start 오름차순으로 정렬하세요.
- masked_text는 raw_text의 각 엔티티 text를 정확히 [TYPE]으로 치환한 결과여야 합니다.
- end는 포함이 아니라 제외(exclusive)입니다.
- STT 특성상 말이 끊기거나 정정되는 표현, 반말/존댓말 혼용, 중복 확인 문구를 허용합니다.
- 일부 번호는 끝자리만 제공되거나, 숫자가 일부만 들리는 형태를 허용합니다.
- 모든 값은 JSON 타입을 지켜 출력하세요(start/end는 정수).

생성 가이드:
- 고객센터 상담, 카드 결제 취소, 이체 확인, 인증번호 확인, 민원 접수, 이메일 회신 등 다양한 맥락을 포함하세요.
- 고객명, 연락처, 카드/계좌 일부번호, 승인번호, 인증번호, 거래 금액, 거래일시, 가맹점명 등이 자연스럽게 포함되어야 합니다.
- 한 건의 데이터는 한 번의 상담 녹취, 채팅 스레드, 이메일 본문, 민원 처리 메모 중 하나처럼 작성하세요.
- 출력 개수는 사용자 요청 수량에 맞추세요.

출력 예시 형식(스키마 참고용):
[
	{
		"raw_text": "상담원: 안녕하세요. 신한카드 고객센터입니다. 고객님 성함이 강민서 맞으실까요? 고객: 네, 010-7712-8840으로 연락 주신다던데요.",
		"entities": [
			{"text": "신한카드", "type": "ORG_NAME", "start": 10, "end": 14},
			{"text": "강민서", "type": "PERSON_NAME", "start": 27, "end": 30},
			{"text": "010-7712-8840", "type": "PHONE_NUMBER", "start": 45, "end": 58}
		],
		"masked_text": "상담원: 안녕하세요. [ORG_NAME] 고객센터입니다. 고객님 성함이 [PERSON_NAME] 맞으실까요? 고객: 네, [PHONE_NUMBER]으로 연락 주신다던데요."
	}
]
