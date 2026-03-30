# !pip install pandas datasets transformers seqeval scikit-learn -q

import ast
import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score


# =========================
# 1. 데이터 준비
# =========================
data = [
    {
        "raw_text": "안녕하세요, 김철수 고객님. 귀하의 카드 번호는 1234-5678-9012-3456입니다.",
        "masked_text": "안녕하세요, [PERSON_NAME] 고객님. 귀하의 카드 번호는 [CARD_NUMBER]입니다.",
        "entities": "[{'text': '김철수', 'type': 'PERSON_NAME', 'start': 7, 'end': 10}, {'text': '1234-5678-9012-3456', 'type': 'CARD_NUMBER', 'start': 28, 'end': 47}]"
    },
    {
        "raw_text": "고객님, 귀하의 전화번호는 010-1234-5678입니다. 확인 부탁드립니다.",
        "masked_text": "고객님, 귀하의 전화번호는 [PHONE_NUMBER]입니다. 확인 부탁드립니다.",
        "entities": "[{'text': '010-1234-5678', 'type': 'PHONE_NUMBER', 'start': 15, 'end': 28}]"
    },
    {
        "raw_text": "홍길동 고객님의 주민등록번호는 123456-1234567입니다.",
        "masked_text": "[PERSON_NAME] 고객님의 주민등록번호는 [RESIDENT_ID]입니다.",
        "entities": "[{'text': '홍길동', 'type': 'PERSON_NAME', 'start': 0, 'end': 3}, {'text': '123456-1234567', 'type': 'RESIDENT_ID', 'start': 17, 'end': 31}]"
    }
]

df = pd.DataFrame(data)
df["entities"] = df["entities"].apply(ast.literal_eval)

print("원본 데이터")
print(df[["raw_text", "masked_text", "entities"]], "\n")


# =========================
# 2. 라벨 정의
# =========================
entity_types = sorted({ent["type"] for ents in df["entities"] for ent in ents})

label_list = ["O"]
for ent_type in entity_types:
    label_list.append(f"B-{ent_type}")
    label_list.append(f"I-{ent_type}")

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

print("라벨 목록")
print(label_list, "\n")


# =========================
# 3. 문자 단위 BIO 라벨 생성 함수
# =========================
def create_char_labels(text, entities):
    char_labels = ["O"] * len(text)
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        ent_type = ent["type"]

        if 0 <= start < len(text):
            char_labels[start] = f"B-{ent_type}"
            for i in range(start + 1, min(end, len(text))):
                char_labels[i] = f"I-{ent_type}"
    return char_labels


# =========================
# 4. Hugging Face Dataset 변환
# =========================
dataset = Dataset.from_pandas(df[["raw_text", "entities"]])

model_checkpoint = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_and_align_labels(example):
    text = example["raw_text"]
    entities = example["entities"]
    char_labels = create_char_labels(text, entities)

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=256,
        return_offsets_mapping=True
    )

    labels = []
    for start, end in tokenized["offset_mapping"]:
        if start == end:
            labels.append(-100)  # special token
        else:
            labels.append(label2id[char_labels[start]])

    tokenized["labels"] = labels
    return tokenized


tokenized_dataset = dataset.map(tokenize_and_align_labels)

remove_cols = [col for col in tokenized_dataset.column_names if col in ["raw_text", "entities", "__index_level_0__"]]
tokenized_dataset = tokenized_dataset.remove_columns(remove_cols)

train_dataset = tokenized_dataset
eval_dataset = tokenized_dataset


# =========================
# 5. 모델 로드
# =========================
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# =========================
# 6. 평가 함수
# =========================
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        cur_preds = []
        cur_labels = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                cur_preds.append(id2label[pred_id])
                cur_labels.append(id2label[label_id])
        true_predictions.append(cur_preds)
        true_labels.append(cur_labels)

    return {
        "f1": f1_score(true_labels, true_predictions)
    }


# =========================
# 7. 학습 설정
# =========================
training_args = TrainingArguments(
    output_dir="./bert_mask_model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# =========================
# 8. 학습
# =========================
trainer.train()


# =========================
# 9. 평가
# =========================
pred_output = trainer.predict(eval_dataset)
print("\n평가 결과")
print(pred_output.metrics)

predictions = np.argmax(pred_output.predictions, axis=2)
labels = pred_output.label_ids

true_predictions = []
true_labels = []

for pred_seq, label_seq in zip(predictions, labels):
    cur_preds = []
    cur_labels = []
    for pred_id, label_id in zip(pred_seq, label_seq):
        if label_id != -100:
            cur_preds.append(id2label[pred_id])
            cur_labels.append(id2label[label_id])
    true_predictions.append(cur_preds)
    true_labels.append(cur_labels)

print("\n분류 리포트")
print(classification_report(true_labels, true_predictions))


# =========================
# 10. 추론 함수
# =========================
def predict_entities(text, model, tokenizer, id2label):
    model.eval()

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    offset_mapping = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**encoded)

    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    entities = []
    current_entity = None

    for pred_id, (start, end) in zip(pred_ids, offset_mapping):
        if start == end:
            continue

        label = id2label[pred_id]

        if label == "O":
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        tag, ent_type = label.split("-", 1)

        if tag == "B":
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {
                "text": text[start:end],
                "type": ent_type,
                "start": start,
                "end": end
            }

        elif tag == "I":
            if current_entity is not None and current_entity["type"] == ent_type:
                current_entity["text"] = text[current_entity["start"]:end]
                current_entity["end"] = end
            else:
                current_entity = {
                    "text": text[start:end],
                    "type": ent_type,
                    "start": start,
                    "end": end
                }

    if current_entity is not None:
        entities.append(current_entity)

    return entities


def mask_text(text, entities):
    masked = text
    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        masked = masked[:ent["start"]] + f"[{ent['type']}]" + masked[ent["end"]:]
    return masked


# =========================
# 11. 테스트
# =========================
test_text = "박영희 고객님의 전화번호는 010-9999-8888이고 카드번호는 1111-2222-3333-4444입니다."

pred_entities = predict_entities(test_text, model, tokenizer, id2label)
pred_masked = mask_text(test_text, pred_entities)

print("\n테스트 문장")
print("원문:", test_text)
print("예측 엔티티:", pred_entities)
print("마스킹 결과:", pred_masked)


# =========================
# 12. 모델 저장
# =========================
trainer.save_model("./bert_mask_model")
tokenizer.save_pretrained("./bert_mask_model")

print("\n모델 저장 완료: ./bert_mask_model")