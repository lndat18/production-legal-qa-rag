"""Single-writer JSONL checkpoints with validated reads and crash recovery."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from pydantic import ValidationError

from production_legal_qa_rag.evaluation.run_models import StageRecord, StageSummary
from production_legal_qa_rag.evaluation.testset_generator import EvalInputError

logger = logging.getLogger(__name__)


class JsonlStore[RecordT: StageRecord]:
    """Own a stage file; last record wins when failed cases are retried."""

    def __init__(self, path: Path, model: type[RecordT]) -> None:
        self.path = path
        self.model = model
        self._lock = threading.Lock()
        self._repair_offset: int | None = None
        self._needs_newline = False
        self.records = self._read()

    @staticmethod
    def key(record: StageRecord) -> tuple[str, str | None]:
        """Include config in identity for shared two-configuration files."""
        return record.case_id, getattr(record, "config", None)

    def _read(self) -> dict[tuple[str, str | None], RecordT]:
        if not self.path.exists():
            return {}
        lines = self.path.read_bytes().splitlines(keepends=True)
        records: dict[tuple[str, str | None], RecordT] = {}
        offset = 0
        for index, line in enumerate(lines):
            try:
                record = self.model.model_validate_json(line)
            except ValidationError, ValueError:
                if index != len(lines) - 1:
                    raise EvalInputError(
                        f"{self.path}: dòng {index + 1} hỏng."
                    ) from None
                logger.warning("Bỏ dòng cuối hỏng: %s", self.path.name)
                self._repair_offset = offset
                break
            records[self.key(record)] = record
            offset += len(line)
        self._needs_newline = bool(
            lines and self._repair_offset is None and not lines[-1].endswith(b"\n")
        )
        return records

    def get(self, identifier: str, config: str | None = None) -> RecordT | None:
        """Return the latest checkpoint for a case/config."""
        return self.records.get((identifier, config))

    def should_run(
        self, identifier: str, *, config: str | None = None, retry_failed: bool = False
    ) -> bool:
        """Skip prior successes and failures unless failed retry was requested."""
        record = self.get(identifier, config)
        return record is None or (retry_failed and record.error is not None)

    def append(self, record: RecordT) -> None:
        """Durably append one validated line, repairing a crashed tail first."""
        record = self.model.model_validate(record.model_dump())
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._repair_offset is not None:
                with self.path.open("r+b") as stream:
                    stream.truncate(self._repair_offset)
                self._repair_offset = None
                self._needs_newline = False
            with self.path.open("ab") as stream:
                if self._needs_newline:
                    stream.write(b"\n")
                    self._needs_newline = False
                stream.write(record.model_dump_json().encode("utf-8") + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.records[self.key(record)] = record

    def summary(
        self,
        identifiers: list[str],
        config: str | None = None,
        *,
        missing_upstream: int = 0,
        quota_exhausted: bool = False,
    ) -> StageSummary:
        """Count latest successes/failures only within the selected testset."""
        records = [self.get(identifier, config) for identifier in identifiers]
        return StageSummary(
            done=sum(r is not None and r.error is None for r in records),
            errors=sum(r is not None and r.error is not None for r in records),
            pending=sum(r is None for r in records),
            missing_upstream=missing_upstream,
            quota_exhausted=quota_exhausted,
        )
