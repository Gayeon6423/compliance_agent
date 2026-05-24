from dotenv import load_dotenv
import os
import requests
import json

# 현재 디렉토리 기준으로 위쪽으로 올라가면서 .env 파일을 찾음
load_dotenv()


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _extract_content(response: dict) -> tuple[str | None, str | None, dict]:
    choices = response.get("choices", []) if isinstance(response, dict) else []
    if not choices:
        raise RuntimeError(f"OpenRouter 응답에 choices가 없습니다: {str(response)[:500]}")

    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None

    # 일부 모델은 content를 list 파트 형태로 반환합니다.
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "".join(text_parts)

    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    return content if isinstance(content, str) else None, finish_reason, message


def chat(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    configured_max_tokens = max_tokens if max_tokens is not None else _get_env_int("MAX_TOKENS", 4096)
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
        "temperature": float(os.getenv("TEMPERATURE", "0.2")),
        "max_tokens": configured_max_tokens,
    }

    last_response = None
    last_status = None
    for attempt in range(3):
        http_response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )

        last_status = http_response.status_code
        try:
            response = http_response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"OpenRouter JSON 파싱 실패 (status={http_response.status_code}): {http_response.text[:500]}"
            ) from exc

        last_response = response

        if http_response.status_code >= 400:
            error_msg = response.get("error", {}).get("message") if isinstance(response, dict) else None
            raise RuntimeError(
                f"OpenRouter 요청 실패 (status={http_response.status_code}): {error_msg or str(response)[:500]}"
            )

        if isinstance(response, dict) and "error" in response:
            error_msg = response.get("error", {}).get("message", "알 수 없는 오류")
            raise RuntimeError(f"OpenRouter API 오류: {error_msg}")

        content, finish_reason, message = _extract_content(response)
        if isinstance(content, str) and content.strip():
            return content.strip()

        if finish_reason == "length" and attempt < 2:
            configured_max_tokens = min(configured_max_tokens * 2, 8192)
            payload["max_tokens"] = configured_max_tokens
            continue

        raise RuntimeError(
            "OpenRouter 응답 content가 비어 있습니다. "
            f"finish_reason={finish_reason}, status={last_status}, message={str(message)[:500]}"
        )

    raise RuntimeError(f"OpenRouter 응답 처리 실패: {str(last_response)[:500]}")

# Test
if __name__ == "__main__":
    system_prompt = "You are a helpful assistant."
    user_prompt = "What's your name? please introduct yourself!"
    print(chat(system_prompt, user_prompt))
