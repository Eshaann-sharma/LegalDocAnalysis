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
        AutoTokenizer,
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
except ModuleNotFoundError:  # pragma: no cover
    AutoConfig = AutoTokenizer = AutoModelForTokenClassification = None  # type: ignore
    DataCollatorForTokenClassification = Trainer = TrainingArguments = None  # type: ignore

from ..inference.labels import ID2LABEL, LABEL2ID, LABELS
from .dataset import load_jsonl, tokenize_and_align
from .metrics import entity_level_prf, per_type_f1


@dataclass(frozen=True)
class FinetuneConfig:
    model_name_or_path: str = "nlpaueb/legal-bert-base-uncased"
    train_path: str = "data/train.jsonl"
    eval_path: str = "data/valid.jsonl"
    output_dir: str = "artifacts/clause_detector"

    max_length: int = 512
    stride: int = 128

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    save_total_limit: int = 2
    seed: int = 42


class _SimpleListDataset:
    """Minimal torch Dataset wrapper around a list of feature dicts."""

    def __init__(self, features: List[Dict[str, Any]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]


def _require_deps() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for fine-tuning. Install it (e.g. pip install torch) "
            "and ensure a GPU build if you want CUDA."
        )
    if AutoConfig is None or AutoTokenizer is None or AutoModelForTokenClassification is None:
        raise ModuleNotFoundError(
            "transformers is required for fine-tuning. Install it (e.g. pip install transformers)."
        )


def _compute_metrics(eval_pred) -> Dict[str, float]:
    """HuggingFace Trainer compute_metrics hook.

    We compute entity-level metrics (BIO spans) rather than token accuracy.
    """
    logits, labels = eval_pred

    # logits: [batch, seq, num_labels]
    # labels: [batch, seq]
    import numpy as np

    pred_ids = np.argmax(logits, axis=-1)

    pred_list: List[List[int]] = pred_ids.tolist()
    gold_list: List[List[int]] = labels.tolist()

    metrics = entity_level_prf(pred_list, gold_list)
    metrics.update(per_type_f1(pred_list, gold_list))

    # Trainer expects plain floats
    return {k: float(v) for k, v in metrics.items()}


def run_finetuning(cfg: FinetuneConfig) -> None:
    """Fine-tune LegalBERT for clause detection token classification.

    Dataset:
      JSONL documents with span annotations (start_char/end_char).

    Outputs:
      - Best checkpoint (by eval f1) loaded at end
      - Saved model/tokenizer into cfg.output_dir
      - Metrics logged per epoch
    """
    _require_deps()

    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)

    train_docs = load_jsonl(cfg.train_path)
    eval_docs = load_jsonl(cfg.eval_path)

    train_features = tokenize_and_align(
        tokenizer=tokenizer,
        examples=train_docs,
        max_length=cfg.max_length,
        stride=cfg.stride,
    )
    eval_features = tokenize_and_align(
        tokenizer=tokenizer,
        examples=eval_docs,
        max_length=cfg.max_length,
        stride=cfg.stride,
    )

    train_ds = _SimpleListDataset(train_features)
    eval_ds = _SimpleListDataset(eval_features)

    config = AutoConfig.from_pretrained(
        cfg.model_name_or_path,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,  # base LegalBERT has no token-classification head
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # GPU support is handled automatically by Trainer when torch/cuda is available.
    # You can verify via: torch.cuda.is_available().
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
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=[],  # keep it simple; add "tensorboard" later if desired
    )

    # transformers v5 uses eval_strategy; v4 uses evaluation_strategy.
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
        compute_metrics=_compute_metrics,
    )

    # transformers v4 Trainer accepts tokenizer; v5 may not.
    if "tokenizer" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    # Save final (best) model + tokenizer.
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    # Minimal CLI via env vars so we don't add extra deps.
    cfg = FinetuneConfig(
        model_name_or_path=os.getenv("CLAUSE_MODEL", "nlpaueb/legal-bert-base-uncased"),
        train_path=os.getenv("CLAUSE_TRAIN", "data/train.jsonl"),
        eval_path=os.getenv("CLAUSE_EVAL", "data/valid.jsonl"),
        output_dir=os.getenv("CLAUSE_OUT", "artifacts/clause_detector"),
    )
    run_finetuning(cfg)
