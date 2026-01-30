"""End-to-end clause extraction pipeline.

Raw text -> clause detection (spans) -> clause classification (type) -> JSON output.
"""

from .end_to_end import EndToEndPipeline, run_pipeline
