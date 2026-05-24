from pathlib import Path
import sys
import pandas as pd
import json
from dotenv import load_dotenv
import os

# 현재 디렉토리 기준으로 위쪽으로 올라가면서 .env 파일을 찾음
load_dotenv()

# 노트북 기준 상위 폴더를 프로젝트 루트로 가정
PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from llm.openrouter import chat
# print("Current Path:", Path.cwd())

# Config
DATA_DIR = PROJECT_ROOT/"v1_text/data"
PROMPT_DIR = PROJECT_ROOT/"v1_text/llm/prompt"
SYSTEM_PROMPT = (PROMPT_DIR/os.getenv("SYSTEM_PROMPT")).read_text(encoding="utf-8")
DATASET_NAME = os.getenv("DATASET_NAME")
TOPIC = os.getenv("TOPIC")
DATA_NUM = int(os.getenv("DATA_NUM"))
OUTPUT_CSV_PATH = DATA_DIR/f"{DATASET_NAME.replace(' ', '_')}.csv"

print(SYSTEM_PROMPT)

# Function
def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _normalize_entities(entities: object) -> str:
    if isinstance(entities, str):
        return entities
    return json.dumps(entities, ensure_ascii=False)


def _parse_json_array(text: str) -> list[dict]:
    cleaned = _strip_json_fence(text)
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("LLM 응답은 JSON 배열(list)이어야 합니다.")
    return data


def _build_user_prompt(dataset_name: str, topic: str, count: int) -> str:
    return f"""
Dataset name: {dataset_name}
Topic: {topic}
Sentence Rows: {count}

Requirements:
- JSON 배열(Array)만 반환하세요. 마크다운 코드펜스 금지.
- 각 원소는 raw_text, entities, masked_text 키를 포함해야 합니다.
- entities는 배열이며 각 원소는 text, type, start, end 키를 포함해야 합니다.
- start/end는 정수여야 합니다.
- 요청한 count만큼만 생성하세요.
""".strip()


def _generate_batch(dataset_name: str, topic: str, batch_count: int, system_prompt: str) -> list[dict]:
    raw_response = chat(
        system_prompt=system_prompt,
        user_prompt=_build_user_prompt(dataset_name, topic, batch_count),
    )
    try:
        return _parse_json_array(raw_response)
    except json.JSONDecodeError as exc:
        if batch_count <= 1:
            raise RuntimeError(
                f"LLM 응답 JSON 파싱 실패: {exc}. 응답 앞부분: {raw_response[:500]}"
            ) from exc

        retry_response = chat(
            system_prompt=system_prompt,
            user_prompt=(
                _build_user_prompt(dataset_name, topic, batch_count)
                + "\n\n중요: 이전 응답은 JSON 파싱에 실패했습니다."
                + "\n유효한 JSON 배열만 출력하고 배열 외 텍스트를 절대 출력하지 마세요."
            ),
        )
        try:
            return _parse_json_array(retry_response)
        except json.JSONDecodeError as retry_exc:
            raise RuntimeError(
                f"LLM 재시도 응답 JSON 파싱 실패: {retry_exc}. 응답 앞부분: {retry_response[:500]}"
            ) from retry_exc

def generate_text(
    dataset_name: str,
    topic: str,
    count: int,
    output_csv_path: str,
    system_prompt: str) -> str:
    system_prompt = system_prompt.strip()
    batch_size = min(3, max(1, count))
    data: list[dict] = []

    while len(data) < count:
        remaining = count - len(data)
        current_batch_size = min(batch_size, remaining)
        batch_data = _generate_batch(dataset_name, topic, current_batch_size, system_prompt)
        data.extend(batch_data[:current_batch_size])
        if current_batch_size == 1:
            batch_size = 1

    print('data:', data)

    expected_keys = {"raw_text", "masked_text", "entities"} 
    normalized_rows = []
    for i, row in enumerate(data[:count]):
        if not isinstance(row, dict):
            raise ValueError(f"{i}번째 항목이 dict가 아닙니다.")

        missing = expected_keys - set(row.keys())
        if missing:
            raise ValueError(f"{i}번째 항목 키 누락: {missing}")

        normalized_rows.append(
            {
                "raw_text": str(row["raw_text"]),
                "masked_text": str(row["masked_text"]),
                "entities": _normalize_entities(row["entities"])
            }
        )

    df = pd.DataFrame(normalized_rows, columns=["raw_text", "masked_text", "entities"])


    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return str(output_path)

# Run
if __name__ == "__main__":

    output_file = generate_text(
        dataset_name=DATASET_NAME,
        topic=TOPIC,
        count=DATA_NUM,
        output_csv_path=OUTPUT_CSV_PATH,
        system_prompt=SYSTEM_PROMPT
    )

    print(f"생성된 파일: {output_file}")
    preview_df = pd.read_csv(output_file)
