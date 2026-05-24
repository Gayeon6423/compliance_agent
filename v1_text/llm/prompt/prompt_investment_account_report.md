증권 계좌 잔고, 보유종목, 수익률, 매매내역, 자산배분을 포함한 투자 보고서를 대상으로 고품질 금융 마스킹 데이터셋을 생성하세요.

반환 형식 규칙(필수):
1) 반드시 JSON 배열(Array)만 반환하세요. 마크다운 코드펜스, 설명 문장, 주석 금지.
2) 배열의 각 원소는 아래 키를 반드시 포함해야 합니다.
- raw_text: 원문 문장 또는 문단 (string)
- entities: 민감정보 엔티티 목록 (array)
- masked_text: 엔티티 타입 토큰으로 치환된 문장 또는 문단 (string)
3) entities의 각 원소는 아래 키를 반드시 포함해야 합니다.
- text: raw_text에 실제로 존재하는 원문 span (string)
- type: 엔티티 타입 (string, 예: INVESTMENT_ACCOUNT_NUMBER, STOCK_NAME, STOCK_CODE, SHARE_COUNT, EVALUATION_AMOUNT, PROFIT_LOSS_AMOUNT, RETURN_RATE, ORDER_NUMBER, CURRENCY_AMOUNT)
- start: raw_text에서 text가 시작하는 문자 인덱스 (0-based, inclusive)
- end: raw_text에서 text가 끝나는 문자 인덱스 (0-based, exclusive)

정합성 규칙(필수):
- raw_text[start:end] == text 를 반드시 만족해야 합니다.
- 같은 문장 또는 문단 내 entities는 start 오름차순으로 정렬하세요.
- masked_text는 raw_text의 각 엔티티 text를 정확히 [TYPE]으로 치환한 결과여야 합니다.
- end는 포함이 아니라 제외(exclusive)입니다.
- 수량, 평가금액, 손익, 수익률, 주문번호, 계좌번호, 종목명, 종목코드 등은 모두 문서 유형상 민감정보로 태깅할 수 있습니다.
- 모든 값은 JSON 타입을 지켜 출력하세요(start/end는 정수).

생성 가이드:
- 실제 금융기관의 자산 보고서, 잔고 확인서, 매매내역서처럼 자연스럽게 작성하세요.
- 국내주식, 해외주식, ETF, 예수금, 외화예수금, 평가손익, 총수익률, 매매일자, 주문번호가 섞일 수 있습니다.
- 문장형, 표형, 요약형 표현을 혼합할 수 있습니다.
- 출력 개수는 사용자 요청 수량에 맞추세요.

출력 예시 형식(스키마 참고용):
[
	{
		"raw_text": "미래에셋증권 종합자산 평가 보고서입니다. 증권계좌번호 270-88-123456의 총 평가금액은 45,820,400원이고 총 수익률은 +7.81%입니다.",
		"entities": [
			{"text": "미래에셋증권", "type": "ISSUER_NAME", "start": 0, "end": 7},
			{"text": "270-88-123456", "type": "INVESTMENT_ACCOUNT_NUMBER", "start": 27, "end": 41},
			{"text": "45,820,400", "type": "EVALUATION_AMOUNT", "start": 50, "end": 60},
			{"text": "+7.81", "type": "RETURN_RATE", "start": 70, "end": 75}
		],
		"masked_text": "[ISSUER_NAME] 종합자산 평가 보고서입니다. 증권계좌번호 [INVESTMENT_ACCOUNT_NUMBER]의 총 평가금액은 [EVALUATION_AMOUNT]원이고 총 수익률은 [RETURN_RATE]%입니다."
	}
]
