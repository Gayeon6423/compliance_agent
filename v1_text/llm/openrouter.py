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
    payload: dict[str, str] = {
        "model": os.getenv("MODEL"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(os.getenv("TEMPERATURE")),
        "max_tokens": int(os.getenv("MAX_TOKENS")),
    }
    response = requests.post(url=url, headers=headers, data=json.dumps(payload)).json()
    content = response["choices"][0]["message"]["content"].strip()
    return content

# Test
if __name__ == "__main__":
    system_prompt = "You are a helpful assistant."
    user_prompt = "What's your name? please introduct yourself!"
    print(chat(system_prompt, user_prompt))
