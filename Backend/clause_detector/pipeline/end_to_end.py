from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from Backend.clause_detector.inference.pipeline import ClauseDetector
from Backend.clause_detector.classifier.inference import ClauseClassifier


@dataclass(frozen=True)
class PipelineConfig:
    # Detection model: token classification checkpoint directory.
    detector_model_path: str = "artifacts/clause_detector"
    # Classifier model: sequence classification checkpoint directory.
    classifier_model_path: str = "artifacts/clause_classifier"


class EndToEndPipeline:
    """Raw text -> detect clause spans -> classify each clause -> JSON."""

    def __init__(
        self,
        *,
        detector_model_path: str,
        classifier_model_path: str,
        detector_device: Optional[str] = None,
        classifier_device: Optional[str] = None,
    ) -> None:
        import os

        if not os.path.exists(detector_model_path):
            raise FileNotFoundError(
                f"Detector checkpoint directory not found: '{detector_model_path}'. "
                "Run token-classification fine-tuning first to create artifacts/clause_detector."
            )
        if not os.path.exists(classifier_model_path):
            raise FileNotFoundError(
                f"Classifier checkpoint directory not found: '{classifier_model_path}'. "
                "Run sequence-classification fine-tuning first to create artifacts/clause_classifier."
            )

        self.detector = ClauseDetector(model_name_or_path=detector_model_path, device=detector_device)
        self.classifier = ClauseClassifier(model_name_or_path=classifier_model_path, device=classifier_device)

    def run(self, text: str) -> Dict[str, Any]:
        """Return JSON-serializable output."""
        if not text or not text.strip():
            return {"clauses": []}

        detected = self.detector.predict(text)

        out_clauses: List[Dict[str, Any]] = []
        for c in detected:
            # Detection span text is used as input to the classifier.
            cls = self.classifier.predict(c.text)
            out_clauses.append(
                {
                    "span": {
                        "start": int(c.start_char),
                        "end": int(c.end_char),
                        "text": c.text,
                        "score": float(c.score),
                    },
                    # If you want to keep detector's coarse type, include it too.
                    "detected_type": c.clause_type,
                    "classification": {
                        "label": cls.label,
                        "score": float(cls.score),
                    },
                }
            )

        return {"clauses": out_clauses}


_DEFAULT_PIPELINE: Optional[EndToEndPipeline] = None


def run_pipeline(
    text: str,
    *,
    detector_model_path: Optional[str] = None,
    classifier_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience API for callers (API routes, scripts, etc.).

    Uses env vars if paths aren't provided:
      - CLAUSE_MODEL_PATH (detector)
      - CLAUSE_CLS_MODEL_PATH (classifier)
    """
    global _DEFAULT_PIPELINE

    detector_path = detector_model_path or os.getenv("CLAUSE_MODEL_PATH", "artifacts/clause_detector")
    classifier_path = classifier_model_path or os.getenv("CLAUSE_CLS_MODEL_PATH", "artifacts/clause_classifier")

    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = EndToEndPipeline(
            detector_model_path=detector_path,
            classifier_model_path=classifier_path,
        )

    return _DEFAULT_PIPELINE.run(text)
