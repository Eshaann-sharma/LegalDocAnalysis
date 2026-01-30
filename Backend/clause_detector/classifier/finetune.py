from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
except ModuleNotFoundError:  # pragma: no cover
    AutoConfig = AutoModelForSequenceClassification = AutoTokenizer = None  # type: ignore
    DataCollatorWithPadding = Trainer = TrainingArguments = None  # type: ignore

from .dataset import build_label_maps, load_jsonl, tokenize_examples
from .metrics import classification_report_per_label


@dataclass(frozen=True)
class ClassifierFinetuneConfig:
    model_name_or_path: str = "nlpaueb/legal-bert-base-uncased"
    train_path: str = "data/clauses_train.jsonl"
    eval_path: str = "data/clauses_valid.jsonl"
    output_dir: str = "artifacts/clause_classifier"

    max_length: int = 256

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    save_total_limit: int = 2
    seed: int = 42


class _SimpleListDataset:
    def __init__(self, features: List[Dict[str, Any]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]


def _require_deps() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for fine-tuning. Install it (e.g. pip install torch)."
        )
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or TrainingArguments is None:
        raise ModuleNotFoundError(
            "transformers is required for fine-tuning. Install it (e.g. pip install transformers)."
        )


def run_finetuning(cfg: ClassifierFinetuneConfig) -> None:
    """Fine-tune LegalBERT for clause type sequence classification.

    Dataset format (JSONL, assumed ready):
      {"clause_id": "...", "text": "<clause text>", "label": "TERMINATION"}

    Outputs:
      - checkpoints saved per epoch
      - best model loaded at end (by macro_f1)
      - final model + tokenizer saved to cfg.output_dir
      - metrics logged per epoch, including per-label precision/recall/f1
    """
    _require_deps()
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_ex = load_jsonl(cfg.train_path)
    eval_ex = load_jsonl(cfg.eval_path)

    if not train_ex:
        raise ValueError(f"No training examples found at {cfg.train_path}")
    if not eval_ex:
        raise ValueError(f"No eval examples found at {cfg.eval_path}")

    label2id, id2label = build_label_maps([e.label for e in (train_ex + eval_ex)])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)

    train_features = tokenize_examples(
        tokenizer=tokenizer,
        examples=train_ex,
        label2id=label2id,
        max_length=cfg.max_length,
    )
    eval_features = tokenize_examples(
        tokenizer=tokenizer,
        examples=eval_ex,
        label2id=label2id,
        max_length=cfg.max_length,
    )

    train_ds = _SimpleListDataset(train_features)
    eval_ds = _SimpleListDataset(eval_features)

    config = AutoConfig.from_pretrained(
        cfg.model_name_or_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        import numpy as np

        pred = np.argmax(logits, axis=-1).tolist()
        gold = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        return classification_report_per_label(gold, pred, id2label)

    import inspect

    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=[],
    )

    if "eval_strategy" in ta_sig:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"

    args = TrainingArguments(**ta_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    if "tokenizer" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    cfg = ClassifierFinetuneConfig(
        model_name_or_path=os.getenv("CLAUSE_CLS_MODEL", "nlpaueb/legal-bert-base-uncased"),
        train_path=os.getenv("CLAUSE_CLS_TRAIN", "data/clauses_train.jsonl"),
        eval_path=os.getenv("CLAUSE_CLS_EVAL", "data/clauses_valid.jsonl"),
        output_dir=os.getenv("CLAUSE_CLS_OUT", "artifacts/clause_classifier"),
    )
    run_finetuning(cfg)
