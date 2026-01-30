"""Clause detection package.

This package is intentionally self-contained so other backend code can import it
without touching non-clause components.

Submodules:
- inference: token-classification span detection
- training: fine-tuning for token classification
- classifier: sequence classification (clause text -> label)
- pipeline: end-to-end (raw text -> spans -> label)
"""

from .inference.pipeline import ClauseDetector, detect_clauses
from .pipeline.end_to_end import EndToEndPipeline, run_pipeline
