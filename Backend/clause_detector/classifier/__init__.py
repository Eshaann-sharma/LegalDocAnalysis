"""Clause type classifier (sequence classification).

Given already-segmented clause text, predict a clause type label.

Includes a HuggingFace Trainer fine-tuning pipeline.
"""

from .finetune import run_finetuning, ClassifierFinetuneConfig
