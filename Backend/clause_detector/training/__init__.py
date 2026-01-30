"""Training utilities for clause detection.

This package provides a HuggingFace Trainer-based fine-tuning entrypoint for
BIO token classification.
"""

from .finetune import run_finetuning
