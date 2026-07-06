from __future__ import annotations

import argparse

try:
    from .training.pipeline import DEFAULT_TEST_TEXT, TrainingConfig, mask_text, predict_entities, train_text_masking_model
except ImportError:
    from training.pipeline import DEFAULT_TEST_TEXT, TrainingConfig, mask_text, predict_entities, train_text_masking_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="금융 텍스트 마스킹 모델 학습 파이프라인")
    parser.add_argument("--train-csv", dest="train_csv_path", default=None, help="학습 CSV 경로")
    parser.add_argument("--eval-csv", dest="eval_csv_path", default=None, help="평가 CSV 경로")
    parser.add_argument("--output-dir", default="bert_mask_model", help="모델 저장 디렉터리")
    parser.add_argument("--model-checkpoint", default="klue/bert-base", help="Hugging Face 모델 체크포인트")
    parser.add_argument("--max-length", type=int, default=256, help="토크나이저 최대 길이")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="학습률")
    parser.add_argument("--train-batch-size", type=int, default=4, help="학습 배치 크기")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="평가 배치 크기")
    parser.add_argument("--epochs", type=int, default=10, help="학습 epoch 수")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--logging-steps", type=int, default=1, help="로깅 스텝 간격")
    parser.add_argument("--test-text", default=DEFAULT_TEST_TEXT, help="학습 후 추론 테스트에 사용할 문장")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        train_csv_path=args.train_csv_path,
        eval_csv_path=args.eval_csv_path,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
    )

    result = train_text_masking_model(config)
    entities = predict_entities(args.test_text, result["model"], result["tokenizer"], result["id2label"], config.max_length)
    masked = mask_text(args.test_text, entities)

    print("평가 결과")
    print(result["metrics"])
    print("\n분류 리포트")
    print(result["report"])
    print("\n테스트 문장")
    print("원문:", args.test_text)
    print("예측 엔티티:", entities)
    print("마스킹 결과:", masked)
    print(f"\n모델 저장 완료: {args.output_dir}")


if __name__ == "__main__":
    main()
