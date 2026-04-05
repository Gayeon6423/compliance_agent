from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_masked_word(value: Any) -> List[Dict[str, str]]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(str(value))


def find_span(raw_text: str, word: str, used_spans: List[tuple[int, int]]) -> tuple[int, int]:
    search_start = 0
    while True:
        start = raw_text.find(word, search_start)
        if start == -1:
            raise ValueError(f"raw_text에서 '{word}'를 찾지 못했습니다.\nraw_text={raw_text}")

        end = start + len(word)
        is_overlapped = any(not (end <= s or start >= e) for s, e in used_spans)
        if not is_overlapped:
            return start, end

        search_start = start + 1


def build_entities(raw_text: str, masked_word: Any) -> List[Dict[str, Any]]:
    items = parse_masked_word(masked_word)
    used_spans: List[tuple[int, int]] = []
    entities: List[Dict[str, Any]] = []

    for item in items:
        word = str(item["word"])
        label = str(item["variable_name"])
        start, end = find_span(raw_text, word, used_spans)
        used_spans.append((start, end))
        entities.append(
            {
                "text": word,
                "label": label,
                "start_raw": start,
                "end_raw": end,
            }
        )

    entities.sort(key=lambda x: x["start_raw"])
    return entities


def build_bio_tags(raw_text: str, entities: List[Dict[str, Any]]) -> List[str]:
    tags = ["O"] * len(raw_text)

    for entity in entities:
        start = entity["start_raw"]
        end = entity["end_raw"]
        label = entity["label"]

        if start < 0 or end > len(raw_text) or start >= end:
            raise ValueError(f"잘못된 span입니다: {entity}")

        tags[start] = f"B-{label}"
        for idx in range(start + 1, end):
            tags[idx] = f"I-{label}"

    return tags


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"raw_text", "masked_text", "masked_word"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["entity"] = df.apply(
        lambda row: build_entities(row["raw_text"], row["masked_word"]),
        axis=1,
    )
    df["bio_tagging"] = df.apply(
        lambda row: build_bio_tags(row["raw_text"], row["entity"]),
        axis=1,
    )
    return df


def main() -> None:
    input_paths = sorted(DATA_DIR.glob("masked_*.csv"))
    if not input_paths:
        raise FileNotFoundError(f"{DATA_DIR} 아래에 masked_*.csv 파일이 없습니다.")

    frames = [pd.read_csv(path) for path in input_paths]
    merged_df = pd.concat(frames, ignore_index=True).drop_duplicates()
    result_df = enrich_dataframe(merged_df)

    output_path = DATA_DIR / "masking_dataset_with_bio.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {output_path}")
    print(result_df[["raw_text", "entity", "bio_tagging"]].head())


if __name__ == "__main__":
    main()
