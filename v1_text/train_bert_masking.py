from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "masking_dataset_with_bio.csv"
MODEL_OUTPUT_DIR = BASE_DIR / "bert_mask_model"
MODEL_NAME = "klue/bert-base"
MAX_LENGTH = 256


def parse_python_literal(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    return ast.literal_eval(str(value))


def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["entity"] = df["entity"].apply(parse_python_literal)
    df["bio_tagging"] = df["bio_tagging"].apply(parse_python_literal)
    return df


def build_label_maps(df: pd.DataFrame) -> tuple[List[str], Dict[str, int], Dict[int, str]]:
    entity_types = sorted(
        {
            entity["label"]
            for entities in df["entity"]
            for entity in entities
        }
    )

    label_list = ["O"]
    for entity_type in entity_types:
        label_list.append(f"B-{entity_type}")
        label_list.append(f"I-{entity_type}")

    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def create_char_labels(raw_text: str, entities: List[Dict[str, Any]]) -> List[str]:
    labels = ["O"] * len(raw_text)
    for entity in entities:
        start = int(entity["start_raw"])
        end = int(entity["end_raw"])
        entity_type = entity["label"]

        labels[start] = f"B-{entity_type}"
        for idx in range(start + 1, end):
            labels[idx] = f"I-{entity_type}"
    return labels


def tokenize_and_align_labels(example: Dict[str, Any], tokenizer, label2id: Dict[str, int]) -> Dict[str, Any]:
    raw_text = example["raw_text"]
    entities = example["entity"]
    char_labels = create_char_labels(raw_text, entities)

    tokenized = tokenizer(
        raw_text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    labels = []
    for start, end in tokenized["offset_mapping"]:
        if start == end:
            labels.append(-100)
            continue

        span_labels = char_labels[start:end]
        token_label = next((label for label in span_labels if label != "O"), "O")
        labels.append(label2id[token_label])

    tokenized["labels"] = labels
    return tokenized


def compute_metrics(eval_pred, id2label: Dict[int, str]) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        pred_tags = []
        true_tags = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            pred_tags.append(id2label[int(pred_id)])
            true_tags.append(id2label[int(label_id)])
        true_predictions.append(pred_tags)
        true_labels.append(true_tags)

    return {"f1": f1_score(true_labels, true_predictions)}


def predict_entities(text: str, model, tokenizer, id2label: Dict[int, str]) -> List[Dict[str, Any]]:
    model.eval()
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    entities: List[Dict[str, Any]] = []
    current = None

    for pred_id, (start, end) in zip(pred_ids, offset_mapping):
        if start == end:
            continue

        label = id2label[int(pred_id)]
        if label == "O":
            if current is not None:
                entities.append(current)
                current = None
            continue

        tag, entity_type = label.split("-", 1)

        if tag == "B":
            if current is not None:
                entities.append(current)
            current = {
                "text": text[start:end],
                "label": entity_type,
                "start_raw": start,
                "end_raw": end,
            }
            continue

        if current is not None and current["label"] == entity_type:
            current["text"] = text[current["start_raw"]:end]
            current["end_raw"] = end
        else:
            current = {
                "text": text[start:end],
                "label": entity_type,
                "start_raw": start,
                "end_raw": end,
            }

    if current is not None:
        entities.append(current)

    return entities


def mask_text_from_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    masked_parts = []
    last_idx = 0

    for entity in sorted(entities, key=lambda x: x["start_raw"]):
        start = entity["start_raw"]
        end = entity["end_raw"]
        label = entity["label"]
        masked_parts.append(text[last_idx:start])
        masked_parts.append(f"[{label}]")
        last_idx = end

    masked_parts.append(text[last_idx:])
    return "".join(masked_parts)


def main() -> None:
    df = load_dataframe()
    label_list, label2id, id2label = build_label_maps(df)

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df[["raw_text", "entity"]], preserve_index=False)
    valid_dataset = Dataset.from_pandas(valid_df[["raw_text", "entity"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id)
    )
    valid_dataset = valid_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id)
    )

    removable_columns = ["raw_text", "entity"]
    train_dataset = train_dataset.remove_columns(removable_columns)
    valid_dataset = valid_dataset.remove_columns(removable_columns)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    trainer.train()

    pred_output = trainer.predict(valid_dataset)
    print(pred_output.metrics)

    predictions = np.argmax(pred_output.predictions, axis=2)
    labels = pred_output.label_ids

    true_predictions = []
    true_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        pred_tags = []
        true_tags = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            pred_tags.append(id2label[int(pred_id)])
            true_tags.append(id2label[int(label_id)])
        true_predictions.append(pred_tags)
        true_labels.append(true_tags)

    print(classification_report(true_labels, true_predictions))

    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    sample_text = "홍길동 고객님의 전화번호는 010-1234-5678입니다."
    sample_entities = predict_entities(sample_text, model, tokenizer, id2label)
    sample_masked_text = mask_text_from_entities(sample_text, sample_entities)

    print("sample_text:", sample_text)
    print("predicted_entities:", sample_entities)
    print("masked_text:", sample_masked_text)


if __name__ == "__main__":
    main()
