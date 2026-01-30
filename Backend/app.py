import os
from typing import List, Optional

# -----------------------
# Flask (existing)
# -----------------------
# Keep the existing /process endpoint if Flask is usable in this environment.
# Some environments may have incompatible Flask/Werkzeug versions; in that case
# we still want FastAPI endpoints to work.
try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS

    from ocr import extract_text
    from summarizer import summarize_text

    # Keep existing behavior for the /process endpoint.
    # NOTE: This imports the legacy prototype-similarity implementation in Backend/clause_detector.py.
    from clause_detector import detect_clauses as legacy_detect_clauses

    app = Flask(__name__)
    CORS(app)

    @app.route("/process", methods=["POST"])
    def process():
        file = request.files["file"]
        filename = file.filename
        file_path = f"temp_{filename}"
        file.save(file_path)

        # 1. OCR
        text = extract_text(file_path)

        # 2. Summarization
        summary = summarize_text(text)

        # 3. Clause Detection (legacy)
        clauses = legacy_detect_clauses(text)

        return jsonify({
            "summary": summary,
            "clauses": clauses,
        })

except Exception:
    app = None


# -----------------------
# FastAPI integration
# -----------------------
# This adds an ASGI app with POST /detect-clauses that uses the fine-tuned
# token-classification pipeline (when dependencies and a checkpoint are available).
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    from asgiref.wsgi import WsgiToAsgi

    from Backend.clause_detector.inference.pipeline import ClauseDetector

    class DetectClausesRequest(BaseModel):
        text: str = Field(..., description="Raw legal document text")

    class DetectedClauseOut(BaseModel):
        type: str
        start: int
        end: int
        text: str
        score: float

    class DetectClausesResponse(BaseModel):
        clauses: List[DetectedClauseOut]

    fastapi_app = FastAPI(title="LegalDocAnalysis API")

    _CLAUSE_DETECTOR: Optional[ClauseDetector] = None

    @fastapi_app.on_event("startup")
    def _load_clause_model() -> None:
        global _CLAUSE_DETECTOR

        model_path = os.getenv("CLAUSE_MODEL_PATH", "artifacts/clause_detector")
        # Load the fine-tuned checkpoint directory (required).
        if not os.path.exists(model_path):
            _CLAUSE_DETECTOR = None
            return

        _CLAUSE_DETECTOR = ClauseDetector(model_name_or_path=model_path)

    @fastapi_app.post("/detect-clauses", response_model=DetectClausesResponse)
    def detect_clauses_endpoint(payload: DetectClausesRequest) -> DetectClausesResponse:
        if _CLAUSE_DETECTOR is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Clause detector not initialized. Set CLAUSE_MODEL_PATH to a fine-tuned checkpoint directory "
                    "(created by Backend.clause_detector.training.finetune)."
                ),
            )

        text = payload.text
        if not text or not text.strip():
            return DetectClausesResponse(clauses=[])

        try:
            clauses = _CLAUSE_DETECTOR.predict(text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Clause detection failed: {e}")

        return DetectClausesResponse(
            clauses=[
                DetectedClauseOut(
                    type=c.clause_type,
                    start=c.start_char,
                    end=c.end_char,
                    text=c.text,
                    score=float(c.score),
                )
                for c in clauses
            ]
        )

    # Expose a single ASGI app that serves both:
    # - existing Flask routes (mounted at /)
    # - new FastAPI endpoints
    if app is not None:
        fastapi_app.mount("/", WsgiToAsgi(app))

    # Uvicorn entrypoint: `uvicorn Backend.app:asgi_app --reload`
    asgi_app = fastapi_app

except ModuleNotFoundError:
    # If FastAPI/asgiref aren't installed, keep Flask-only behavior.
    asgi_app = None


if __name__ == "__main__":
    # Flask dev server (kept for backwards compatibility).
    # For FastAPI + Flask together, run via Uvicorn:
    #   uvicorn Backend.app:asgi_app --reload
    if app is None:
        raise RuntimeError(
            "Flask app could not be initialized in this environment. "
            "Run the ASGI app with Uvicorn instead (Backend.app:asgi_app) after installing FastAPI dependencies."
        )
    app.run(debug=True)
