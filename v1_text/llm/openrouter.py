from dotenv import load_dotenv
import os
import requests
import json

# 현재 디렉토리 기준으로 위쪽으로 올라가면서 .env 파일을 찾음
load_dotenv()


def chat(system_prompt: str, user_prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-OpenRouter-Title": "complianceAgent",
    }
    payload: dict[str, object] = {
        "model": os.getenv("MODEL"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(os.getenv("TEMPERATURE")),
        "max_tokens": int(os.getenv("MAX_TOKENS")),
    }

    http_response = requests.post(
        url=url,
        headers=headers,
        data=json.dumps(payload),
        timeout=60,
    )

    try:
        response = http_response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"OpenRouter JSON 파싱 실패 (status={http_response.status_code}): {http_response.text[:500]}"
        ) from exc

    if http_response.status_code >= 400:
        error_msg = response.get("error", {}).get("message") if isinstance(response, dict) else None
        raise RuntimeError(
            f"OpenRouter 요청 실패 (status={http_response.status_code}): {error_msg or str(response)[:500]}"
        )

    if isinstance(response, dict) and "error" in response:
        error_msg = response.get("error", {}).get("message", "알 수 없는 오류")
        raise RuntimeError(f"OpenRouter API 오류: {error_msg}")

    choices = response.get("choices", []) if isinstance(response, dict) else []
    if not choices:
        raise RuntimeError(f"OpenRouter 응답에 choices가 없습니다: {str(response)[:500]}")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else None

    # 일부 모델은 content를 list 파트 형태로 반환합니다.
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "".join(text_parts)

    if not isinstance(content, str) or not content.strip():
        finish_reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
        raise RuntimeError(
            "OpenRouter 응답 content가 비어 있습니다. "
            f"finish_reason={finish_reason}, message={str(message)[:500]}"
        )

    return content.strip()

# Test
if __name__ == "__main__":
    system_prompt = "You are a helpful assistant."
    user_prompt = "What's your name? please introduct yourself!"
    print(chat(system_prompt, user_prompt))
