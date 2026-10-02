"""Validated contracts for resumable evaluation stages (spec section 11)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.testset_generator import EvalInputError
from production_legal_qa_rag.generation.models import Citation, Usage
from production_legal_qa_rag.retrieval.models import PrecomputedQuery, RetrievedChunk

type RetrievalConfig = Literal["mmr_on", "mmr_off"]
CONFIGS: tuple[RetrievalConfig, ...] = ("mmr_on", "mmr_off")
SCORING_BATCH_SIZE = 10


def case_id(user_input: str) -> str:
    """Return the stable 12-hex case identifier of the unchanged question."""
    return hashlib.sha256(user_input.encode("utf-8")).hexdigest()[:12]


class EvalConfig(BaseModel):
    """CLI run scope; settings are independent from production configuration."""

    testset: Path = Path("data/eval/golden_testset.json")
    output_dir: Path = Path("data/eval/phase2")
    limit: int | None = Field(default=None, gt=0)
    retry_failed: bool = False
    workers: int = Field(default=9, ge=1, le=9)
    config: RetrievalConfig | None = None


class StageRecord(BaseModel):
    """Common checkpoint identity and sanitized error metadata."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    case_id: str = Field(pattern=r"^[0-9a-f]{12}$")
    error: str | None = None


class HydeRecord(StageRecord):
    """HyDE null is a successful production fallback, separate from error."""

    hypothetical_document: str | None = None


class EmbeddingRecord(StageRecord):
    """Reusable dense preparation for both retrieval configurations."""

    precomputed: PrecomputedQuery | None = None


class RetrievalRecord(StageRecord):
    """Ranked retrieval result or explicit fallback/no-context failure."""

    config: RetrievalConfig
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class AnswerRecord(StageRecord):
    """Verified pipeline events reduced to one resumable answer checkpoint."""

    config: RetrievalConfig
    outcome: Literal["answered", "insufficient_evidence", "unable_to_verify", "error"]
    response: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    repair_used: bool = False
    warning_codes: list[str] = Field(default_factory=list)
    error_code: str | None = None
    usage: Usage | None = None
    prompt_version: str


class ScoreRecord(StageRecord):
    """Metric scores; failed or non-finite values are null and retriable."""

    config: RetrievalConfig
    scores: dict[str, float | None] = Field(default_factory=dict)
    reused_from: RetrievalConfig | None = None


class StageSummary(BaseModel):
    """Progress statistics returned without reading any external service."""

    done: int = 0
    errors: int = 0
    pending: int = 0
    missing_upstream: int = 0
    quota_exhausted: bool = False


def load_testset(config: EvalConfig) -> list[GoldenTestCase]:
    """Validate a testset before creating clients; limit selects its prefix."""
    try:
        raw = json.loads(config.testset.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise TypeError
        cases = [GoldenTestCase.model_validate(row) for row in raw]
    except OSError, UnicodeError, ValueError, TypeError, ValidationError:
        raise EvalInputError("Testset thiếu hoặc sai schema JSON.") from None
    identifiers = [case_id(case.user_input) for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise EvalInputError("Testset có case_id trùng.")
    if any(case.empty_required_fields() for case in cases):
        raise EvalInputError("Testset có trường bắt buộc rỗng.")
    return cases[: config.limit]
