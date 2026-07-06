from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

IGNORE_LABEL_ID = -100
DEFAULT_TEST_TEXT = "박영희 고객님의 전화번호는 010-9999-8888이고 카드번호는 1111-2222-3333-4444입니다."
SAMPLE_RECORDS: list[dict[str, Any]] = [
    {
        "raw_text": "안녕하세요, 김철수 고객님. 귀하의 카드 번호는 1234-5678-9012-3456입니다.",
        "masked_text": "안녕하세요, [PERSON_NAME] 고객님. 귀하의 카드 번호는 [CARD_NUMBER]입니다.",
        "entities": [
            {"text": "김철수", "type": "PERSON_NAME", "start": 7, "end": 10},
            {"text": "1234-5678-9012-3456", "type": "CARD_NUMBER", "start": 28, "end": 47},
        ],
    },
    {
        "raw_text": "고객님, 귀하의 전화번호는 010-1234-5678입니다. 확인 부탁드립니다.",
        "masked_text": "고객님, 귀하의 전화번호는 [PHONE_NUMBER]입니다. 확인 부탁드립니다.",
        "entities": [
            {"text": "010-1234-5678", "type": "PHONE_NUMBER", "start": 15, "end": 28},
        ],
    },
    {
        "raw_text": "홍길동 고객님의 주민등록번호는 123456-1234567입니다.",
        "masked_text": "[PERSON_NAME] 고객님의 주민등록번호는 [RESIDENT_REGISTRATION_NUMBER]입니다.",
        "entities": [
            {"text": "홍길동", "type": "PERSON_NAME", "start": 0, "end": 3},
            {"text": "123456-1234567", "type": "RESIDENT_REGISTRATION_NUMBER", "start": 17, "end": 31},
        ],
    },
]


@dataclass(slots=True)
class TrainingConfig:
    model_checkpoint: str = "klue/bert-base"
    output_dir: str = "bert_mask_model"
    train_csv_path: str | None = None
    eval_csv_path: str | None = None
    max_length: int = 256
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    logging_steps: int = 1


def _parse_entities(raw_value: object) -> list[dict[str, Any]]:
    if isinstance(raw_value, list):
        return raw_value
    if not isinstance(raw_value, str):
        raise TypeError("entities는 문자열 또는 배열이어야 합니다.")

    raw_value = raw_value.strip()
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(raw_value)

    if not isinstance(parsed, list):
        raise ValueError("entities는 배열이어야 합니다.")
    return parsed


def load_records(csv_path: str | Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        frame = pd.DataFrame(SAMPLE_RECORDS)
    else:
        frame = pd.read_csv(csv_path)

    required_columns = {"raw_text", "entities"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")

    normalized = frame.copy()
    normalized["entities"] = normalized["entities"].apply(_parse_entities)
    if "masked_text" not in normalized.columns:
        normalized["masked_text"] = ""
    return normalized


def load_training_frames(config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = load_records(config.train_csv_path)
    eval_frame = load_records(config.eval_csv_path) if config.eval_csv_path else train_frame.copy()
    return train_frame, eval_frame


def build_label_maps(frames: list[pd.DataFrame]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    entity_types = sorted(
        {
            entity["type"]
            for frame in frames
            for entities in frame["entities"]
            for entity in entities
        }
    )
    label_list = ["O"]
    for entity_type in entity_types:
        label_list.append(f"B-{entity_type}")
        label_list.append(f"I-{entity_type}")
    label2id = {label: index for index, label in enumerate(label_list)}
    id2label = {index: label for label, index in label2id.items()}
    return label_list, label2id, id2label


def create_char_labels(text: str, entities: list[dict[str, Any]]) -> list[str]:
    labels = ["O"] * len(text)
    for entity in entities:
        start = int(entity["start"])
        end = int(entity["end"])
        entity_type = str(entity["type"])
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"잘못된 span 범위입니다: {entity}")
        if text[start:end] != entity["text"]:
            raise ValueError(f"span text 불일치: {entity}")
        labels[start] = f"B-{entity_type}"
        for index in range(start + 1, end):
            labels[index] = f"I-{entity_type}"
    return labels


def _tokenize_and_align_labels(example: dict[str, Any], tokenizer, label2id: dict[str, int], max_length: int) -> dict[str, Any]:
    text = example["raw_text"]
    entities = example["entities"]
    char_labels = create_char_labels(text, entities)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    labels = []
    for start, end in tokenized["offset_mapping"]:
        if start == end:
            labels.append(IGNORE_LABEL_ID)
        else:
            labels.append(label2id[char_labels[start]])

    tokenized["labels"] = labels
    return tokenized


def prepare_tokenized_dataset(frame: pd.DataFrame, tokenizer, label2id: dict[str, int], max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(frame[["raw_text", "entities"]], preserve_index=False)
    tokenized = dataset.map(
        lambda example: _tokenize_and_align_labels(example, tokenizer, label2id, max_length)
    )
    removable_columns = [
        column
        for column in tokenized.column_names
        if column in {"raw_text", "entities", "offset_mapping", "__index_level_0__"}
    ]
    return tokenized.remove_columns(removable_columns)


def build_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        prediction_ids = np.argmax(predictions, axis=2)

        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []
        for predicted_sequence, label_sequence in zip(prediction_ids, labels):
            filtered_predictions: list[str] = []
            filtered_labels: list[str] = []
            for predicted_id, label_id in zip(predicted_sequence, label_sequence):
                if label_id == IGNORE_LABEL_ID:
                    continue
                filtered_predictions.append(id2label[int(predicted_id)])
                filtered_labels.append(id2label[int(label_id)])
            true_predictions.append(filtered_predictions)
            true_labels.append(filtered_labels)

        return {"f1": float(f1_score(true_labels, true_predictions))}

    return compute_metrics


def build_trainer(
    config: TrainingConfig,
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    id2label: dict[int, str],
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(id2label),
    )


def evaluate_predictions(prediction_output, id2label: dict[int, str]) -> tuple[dict[str, float], str]:
    prediction_ids = np.argmax(prediction_output.predictions, axis=2)
    label_ids = prediction_output.label_ids
    true_predictions: list[list[str]] = []
    true_labels: list[list[str]] = []

    for predicted_sequence, label_sequence in zip(prediction_ids, label_ids):
        filtered_predictions: list[str] = []
        filtered_labels: list[str] = []
        for predicted_id, label_id in zip(predicted_sequence, label_sequence):
            if label_id == IGNORE_LABEL_ID:
                continue
            filtered_predictions.append(id2label[int(predicted_id)])
            filtered_labels.append(id2label[int(label_id)])
        true_predictions.append(filtered_predictions)
        true_labels.append(filtered_labels)

    metrics = {key: float(value) for key, value in prediction_output.metrics.items() if isinstance(value, (int, float))}
    report = classification_report(true_labels, true_predictions)
    return metrics, report


def predict_entities(text: str, model, tokenizer, id2label: dict[int, str], max_length: int = 256) -> list[dict[str, Any]]:
    model.eval()
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**encoded)

    prediction_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    entities: list[dict[str, Any]] = []
    current_entity: dict[str, Any] | None = None

    for prediction_id, (start, end) in zip(prediction_ids, offset_mapping):
        if start == end:
            continue

        label = id2label[int(prediction_id)]
        if label == "O":
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        tag, entity_type = label.split("-", 1)
        if tag == "B":
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {
                "text": text[start:end],
                "type": entity_type,
                "start": start,
                "end": end,
            }
        elif current_entity is not None and current_entity["type"] == entity_type:
            current_entity["text"] = text[current_entity["start"]:end]
            current_entity["end"] = end
        else:
            current_entity = {
                "text": text[start:end],
                "type": entity_type,
                "start": start,
                "end": end,
            }

    if current_entity is not None:
        entities.append(current_entity)
    return entities


def mask_text(text: str, entities: list[dict[str, Any]]) -> str:
    masked = text
    for entity in sorted(entities, key=lambda item: int(item["start"]), reverse=True):
        masked = masked[: entity["start"]] + f"[{entity['type']}]" + masked[entity["end"] :]
    return masked


def train_text_masking_model(config: TrainingConfig) -> dict[str, Any]:
    train_frame, eval_frame = load_training_frames(config)
    label_list, label2id, id2label = build_label_maps([train_frame, eval_frame])

    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
    train_dataset = prepare_tokenized_dataset(train_frame, tokenizer, label2id, config.max_length)
    eval_dataset = prepare_tokenized_dataset(eval_frame, tokenizer, label2id, config.max_length)

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    trainer = build_trainer(config, model, tokenizer, train_dataset, eval_dataset, id2label)
    trainer.train()

    prediction_output = trainer.predict(eval_dataset)
    metrics, report = evaluate_predictions(prediction_output, id2label)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return {
        "trainer": trainer,
        "model": model,
        "tokenizer": tokenizer,
        "label2id": label2id,
        "id2label": id2label,
        "metrics": metrics,
        "report": report,
        "train_frame": train_frame,
        "eval_frame": eval_frame,
    }
