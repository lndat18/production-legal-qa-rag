"""Dịch mẫu tiếng Anh trong golden testset raw qua Google Apps Script (evaluation_spec.md mục 12).

`gpt-oss-120b` đôi khi trả `user_input`/`reference` bằng tiếng Anh dù đã ép prompt tiếng
Việt. Module này phát hiện các trường tiếng Anh (hàm thuần, theo tỉ lệ từ có dấu tiếng Việt),
dịch từng trường qua web app Apps Script của người dùng, kiểm bằng code rằng tham chiếu pháp
lý (Điều/Khoản/Điểm/Chương/Mục) và số dòng không đổi, rồi ghi đè tại chỗ vào raw kèm bản gốc
(`original_en`) và cờ soát tay (`translation_review`).

Không import `ragas`: chạy được trên venv thường. `requests` chỉ import lười khi gọi mạng thật
(nhóm `eval`), nên `--dry-run` và toàn bộ hàm thuần không cần nó. Log chỉ có số đếm, số dòng và
tên loại lỗi — không URL, key, nội dung hay thông điệp exception (mục 12.2).
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.testset_generator import (
    DEFAULT_OUTPUT_DIR,
    RAW_TESTSET_FILENAME,
    EvalInputError,
    _describe_traceback,
    read_raw_rows,
    write_raw_rows,
)

logger = logging.getLogger(__name__)

# Hằng số nội bộ (không phải env var, như GENERATE_SIZE). Ngưỡng chốt sau khi xem phân bố
# thật từ `translate --dry-run` (mục 12.3).
ENGLISH_MAX_DIACRITIC_RATIO: Final = 0.10
AMBIGUOUS_RATIO_RANGE: Final = (0.05, 0.30)
MIN_LETTER_TOKENS: Final = 4
# Biên an toàn dưới 6.400 ký tự đã đo chạy ổn qua GET (mục 12.2).
MAX_REQUEST_CHARS: Final = 4_000
MAX_ATTEMPTS: Final = 3
MAX_CONSECUTIVE_TRANSLATE_ERRORS: Final = 3
_BACKOFF_BASE_SECONDS: Final = 1.0
GLOSSARY_FILENAME: Final = "translation_glossary.json"

REVIEW_CITATION_MISMATCH: Final = "citation_mismatch"
REVIEW_LINE_COUNT_MISMATCH: Final = "line_count_mismatch"
REVIEW_STILL_ENGLISH: Final = "still_english"
REVIEW_TOO_LONG: Final = "too_long"
REVIEW_TRANSLATE_ERROR: Final = "translate_error"
REVIEW_CONTEXTS_TRANSLATED: Final = "contexts_translated"

FieldName = Literal["user_input", "reference", "reference_contexts"]


class TranslatorConfigError(EvalInputError):
    """Lỗi cấu hình dịch (URL sai, `forbidden`, trang HTML quyền): không retry, mã thoát 2."""


class TranslateError(RuntimeError):
    """Lỗi dịch tạm thời đã hết retry; thông điệp chỉ nêu loại lỗi, không URL/nội dung."""


class FieldRejected(Exception):
    """Bản dịch một trường không đạt kiểm tra bất biến; `reason` là mã `translation_review`."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _TransientResponseError(Exception):
    """Phản hồi lỗi tạm thời (5xx, JSON hỏng, thiếu `translatedText`...) — được retry."""


class TranslateSettings(BaseSettings):
    """Cấu hình Apps Script dịch (evaluation_spec.md mục 12.2).

    `env_ignore_empty=True`: `TRANSLATE_URL=`/`TRANSLATE_KEY=` để trống trong `.env.example`
    coi như chưa đặt (URL thiếu -> lỗi nêu tên biến; key thiếu -> không gửi tham số `key`).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    url: str = Field(min_length=1, validation_alias="TRANSLATE_URL")
    key: str | None = Field(default=None, validation_alias="TRANSLATE_KEY")
    timeout_seconds: int = 60


def load_translate_settings() -> TranslateSettings:
    """Nạp `TranslateSettings` từ môi trường/`.env`.

    Raises:
        EvalInputError: Thiếu `TRANSLATE_URL`. Thông báo chỉ nêu TÊN biến, không giá trị.
    """
    try:
        return TranslateSettings()
    except ValidationError:
        raise EvalInputError(
            "Thiếu hoặc để trống TRANSLATE_URL trong .env (URL web app Google Apps Script)."
        ) from None


# ---------------------------------------------------------------------------
# Phát hiện trường tiếng Anh (mục 12.3) — hàm thuần
# ---------------------------------------------------------------------------


class FieldMeasure(BaseModel):
    """Tỉ lệ từ có dấu tiếng Việt của một trường đủ dài để phân loại."""

    field: FieldName
    index: int | None = None  # chỉ số phần tử khi `field == "reference_contexts"`
    ratio: float
    chars: int

    @property
    def label(self) -> str:
        """Tên hiển thị, vd `reference_contexts[1]`."""
        return self.field if self.index is None else f"{self.field}[{self.index}]"

    @property
    def is_english(self) -> bool:
        """`True` khi tỉ lệ dưới `ENGLISH_MAX_DIACRITIC_RATIO`."""
        return self.ratio < ENGLISH_MAX_DIACRITIC_RATIO


def _has_vietnamese_diacritic(token: str) -> bool:
    return any(
        char in "đĐ"
        or any(
            unicodedata.combining(part) for part in unicodedata.normalize("NFD", char)
        )
        for char in token
    )


def diacritic_ratio(text: str) -> float | None:
    """Tỉ lệ token chữ có ký tự dấu/`đ` tiếng Việt; `None` nếu < `MIN_LETTER_TOKENS` token chữ."""
    tokens = [token for token in text.split() if any(c.isalpha() for c in token)]
    if len(tokens) < MIN_LETTER_TOKENS:
        return None
    return sum(_has_vietnamese_diacritic(token) for token in tokens) / len(tokens)


def is_english(text: str) -> bool:
    """`True` khi `text` đủ dài để phân loại và tỉ lệ dấu tiếng Việt dưới ngưỡng."""
    ratio = diacritic_ratio(text)
    return ratio is not None and ratio < ENGLISH_MAX_DIACRITIC_RATIO


def measure_fields(case: GoldenTestCase) -> list[FieldMeasure]:
    """Đo tỉ lệ dấu từng trường (độc lập nhau); bỏ qua trường quá ngắn để phân loại."""
    candidates: list[tuple[FieldName, int | None, str]] = [
        ("user_input", None, case.user_input),
        ("reference", None, case.reference),
    ]
    candidates.extend(
        ("reference_contexts", index, context)
        for index, context in enumerate(case.reference_contexts)
    )
    measures = []
    for field, index, text in candidates:
        ratio = diacritic_ratio(text)
        if ratio is not None:
            measures.append(
                FieldMeasure(field=field, index=index, ratio=ratio, chars=len(text))
            )
    return measures


def detect_english_fields(case: GoldenTestCase) -> list[FieldMeasure]:
    """Các trường của `case` được coi là tiếng Anh (cần dịch), theo thứ tự trường."""
    return [measure for measure in measure_fields(case) if measure.is_english]


# ---------------------------------------------------------------------------
# Tách, chuẩn hoá, bảng thuật ngữ, kiểm bất biến (mục 12.4) — hàm thuần
# ---------------------------------------------------------------------------


def split_for_request(text: str, max_chars: int = MAX_REQUEST_CHARS) -> list[str]:
    """Tách `text` thành các phần ≤ `max_chars`, gộp theo dòng; nối lại bằng `\\n` ra đúng `text`.

    Raises:
        FieldRejected: Một dòng đơn dài hơn `max_chars` (`too_long`) — không dịch.
    """
    parts: list[str] = []
    current: list[str] = []
    current_chars = 0
    for line in text.split("\n"):
        if len(line) > max_chars:
            raise FieldRejected(REVIEW_TOO_LONG)
        added = len(line) + (1 if current else 0)
        if current and current_chars + added > max_chars:
            parts.append("\n".join(current))
            current, current_chars, added = [], 0, len(line)
        current.append(line)
        current_chars += added
    parts.append("\n".join(current))
    return parts


_HORIZONTAL_SPACES: Final = re.compile(r"[^\S\n]+")
_SPACE_BEFORE_PUNCTUATION: Final = re.compile(r" +([,.;:])")
_GLUED_WORDS: Final = re.compile(r"([a-zà-ỹ])([A-ZĐ])")


def normalize_vietnamese(text: str) -> str:
    """Chuẩn hoá nhẹ bản dịch: NFC, gộp khoảng trắng, bỏ dấu cách trước `,.;:`, tách chữ dính hoa.

    Cố ý KHÔNG sửa "dính chữ" kiểu "vệsức" bằng regex/từ điển (mục 4.1, 12.4): để bước soát tay.
    """
    text = unicodedata.normalize("NFC", text)
    text = _HORIZONTAL_SPACES.sub(" ", text)
    text = "\n".join(line.strip(" ") for line in text.split("\n"))
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    return _GLUED_WORDS.sub(r"\1 \2", text)


def _match_case_of_first_letter(matched: str, replacement: str) -> str:
    if matched[:1].isupper() and replacement:
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply_glossary(text: str, glossary: Mapping[str, str]) -> str:
    """Thay thuật ngữ trên BẢN DỊCH: khớp không phân biệt hoa/thường, cụm dài trước, giữ hoa đầu."""
    normalized = {
        unicodedata.normalize("NFC", wrong).casefold(): right
        for wrong, right in glossary.items()
        if wrong.strip()
    }
    if not normalized:
        return text
    pattern = re.compile(
        "|".join(
            re.escape(wrong) for wrong in sorted(normalized, key=len, reverse=True)
        ),
        re.IGNORECASE,
    )
    return pattern.sub(
        lambda match: _match_case_of_first_letter(
            match.group(0), normalized[match.group(0).casefold()]
        ),
        text,
    )


def load_glossary(path: Path) -> dict[str, str]:
    """Đọc bảng thuật ngữ `{"sai": "đúng"}`; thiếu file = bảng rỗng.

    Raises:
        EvalInputError: File không phải UTF-8/JSON hợp lệ dạng `{chuỗi: chuỗi}`.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EvalInputError(f"{path} không phải JSON UTF-8 hợp lệ: {error}") from error
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise EvalInputError(f"{path} phải là đối tượng JSON dạng {{chuỗi: chuỗi}}.")
    return data


_ENGLISH_REFERENCE: Final = re.compile(
    r"\b(Article|Clause|Point|Chapter|Section|Paragraph)s?\s+([0-9]+|[IVXLC]+\b|[A-Za-z]\b)",
    re.IGNORECASE,
)
_VIETNAMESE_REFERENCE: Final = re.compile(
    r"\b(Điều|Khoản|Điểm|Chương|Mục)\s+([0-9]+|[IVXLC]+\b|[A-Za-zĐđ]\b)",
    re.IGNORECASE,
)
# `Paragraph` không có trong bảng ánh xạ của spec; dịch pháp lý thường là Khoản.
_REFERENCE_KIND_TO_VIETNAMESE: Final = {
    "article": "điều",
    "clause": "khoản",
    "paragraph": "khoản",
    "point": "điểm",
    "chapter": "chương",
    "section": "mục",
}


def extract_legal_references(
    text: str, *, language: Literal["en", "vi"]
) -> set[tuple[str, str]]:
    """Tập cặp `(loại, số/chữ)` tham chiếu pháp lý, đã quy về loại tiếng Việt viết thường.

    Với `language="en"` ánh xạ `Article→điều`, `Clause→khoản`, `Point→điểm`,
    `Chapter→chương`, `Section→mục` (và `Paragraph→khoản`); so tập, không so số lần.
    """
    text = unicodedata.normalize("NFC", text)
    if language == "en":
        return {
            (_REFERENCE_KIND_TO_VIETNAMESE[kind.lower()], value.lower())
            for kind, value in _ENGLISH_REFERENCE.findall(text)
        }
    return {
        (kind.lower(), value.lower())
        for kind, value in _VIETNAMESE_REFERENCE.findall(text)
    }


def _count_non_empty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def check_translation_invariants(original: str, translated: str) -> None:
    """Kiểm bất biến của bản dịch một trường (mục 12.4 bước 5).

    Raises:
        FieldRejected: `still_english` (rỗng hoặc vẫn dưới ngưỡng dấu),
            `citation_mismatch` hoặc `line_count_mismatch`.
    """
    if not translated.strip() or is_english(translated):
        raise FieldRejected(REVIEW_STILL_ENGLISH)
    english_references = extract_legal_references(original, language="en")
    if english_references and english_references != extract_legal_references(
        translated, language="vi"
    ):
        raise FieldRejected(REVIEW_CITATION_MISMATCH)
    if _count_non_empty_lines(original) != _count_non_empty_lines(translated):
        raise FieldRejected(REVIEW_LINE_COUNT_MISMATCH)


# ---------------------------------------------------------------------------
# Gọi Apps Script (mục 12.2, 12.4 bước 2)
# ---------------------------------------------------------------------------


class TextTranslator(Protocol):
    """Đối tượng dịch một đoạn văn bản Anh -> Việt (Apps Script thật, hoặc giả lập trong test)."""

    def translate(self, text: str) -> str:
        """Trả bản dịch tiếng Việt của `text`."""
        ...


class _HttpResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def content(self) -> bytes: ...


_HttpGet = Callable[[str, Mapping[str, str], float], _HttpResponse]


def _requests_get(url: str, params: Mapping[str, str], timeout: float) -> _HttpResponse:
    try:
        import requests
    except ImportError:
        raise TranslatorConfigError(
            "Thiếu thư viện requests; chạy với --group eval --no-group production."
        ) from None
    return requests.get(url, params=dict(params), timeout=timeout)


class AppsScriptTranslator:
    """Dịch qua Google Apps Script web app: `GET ?text=&source=en&target=vi[&key=]`.

    `get`, `sleep` và `jitter` tiêm được để test không cần mạng hay chờ thật. Mọi thông
    điệp lỗi chỉ nêu loại lỗi (không URL/key/nội dung).
    """

    def __init__(
        self,
        settings: TranslateSettings,
        *,
        get: _HttpGet = _requests_get,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._settings = settings
        self._get = get
        self._sleep = sleep
        self._jitter = jitter

    def translate(self, text: str) -> str:
        """Dịch `text`, retry tối đa `MAX_ATTEMPTS` lần cho lỗi tạm thời.

        Raises:
            TranslatorConfigError: `forbidden`, HTML quyền/đăng nhập, HTTP 4xx (không retry).
            TranslateError: Hết retry vì timeout/kết nối/5xx/phản hồi không hợp lệ.
        """
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._request_once(text)
            except (OSError, _TransientResponseError) as error:
                if attempt == MAX_ATTEMPTS - 1:
                    raise TranslateError(
                        f"Dịch thất bại sau {MAX_ATTEMPTS} lần: {type(error).__name__}"
                    ) from None
                self._sleep(_BACKOFF_BASE_SECONDS * 2**attempt + self._jitter())
        raise AssertionError("unreachable")  # pragma: no cover

    def _request_once(self, text: str) -> str:
        params = {"text": text, "source": "en", "target": "vi"}
        if self._settings.key:
            params["key"] = self._settings.key
        response = self._get(self._settings.url, params, self._settings.timeout_seconds)
        return _parse_response(response)


def _parse_response(response: _HttpResponse) -> str:
    status = response.status_code
    if status >= 500 or status == 429:
        raise _TransientResponseError(f"HTTP {status}")
    if status >= 400:
        raise TranslatorConfigError(
            f"Apps Script trả HTTP {status}; kiểm tra TRANSLATE_URL và quyền truy cập."
        )
    try:
        body = response.content.decode("utf-8")
    except UnicodeDecodeError:
        raise _TransientResponseError("phản hồi không phải UTF-8") from None
    if body.lstrip().startswith("<"):
        raise TranslatorConfigError(
            "Apps Script trả trang HTML (đăng nhập/quyền/quota); kiểm tra deploy "
            "(Who has access: Anyone) và TRANSLATE_URL."
        )
    try:
        payload = json.loads(body)
    except ValueError:
        raise _TransientResponseError("phản hồi không phải JSON") from None
    if not isinstance(payload, dict):
        raise _TransientResponseError("JSON không phải đối tượng")
    error = payload.get("error")
    if isinstance(error, str) and error.strip().lower() == "forbidden":
        raise TranslatorConfigError(
            "Apps Script từ chối (forbidden); TRANSLATE_KEY sai hoặc thiếu."
        )
    if error is not None:
        raise _TransientResponseError("Apps Script báo lỗi")
    translated = payload.get("translatedText")
    if not isinstance(translated, str) or not translated.strip():
        raise _TransientResponseError("thiếu translatedText")
    return translated


# ---------------------------------------------------------------------------
# Dịch một trường / một mẫu (mục 12.4, 12.5)
# ---------------------------------------------------------------------------


def translate_field(
    text: str, translator: TextTranslator, glossary: Mapping[str, str]
) -> str:
    """Dịch một trường theo đúng thứ tự tách -> dịch -> chuẩn hoá -> thuật ngữ -> kiểm bất biến.

    Raises:
        FieldRejected: Dòng quá dài hoặc bản dịch trượt kiểm bất biến (không có bản dịch hợp lệ).
        TranslateError: Lỗi mạng tạm thời đã hết retry.
        TranslatorConfigError: Lỗi cấu hình (không retry).
    """
    parts = split_for_request(text)
    translated = "\n".join(translator.translate(part) for part in parts)
    translated = apply_glossary(normalize_vietnamese(translated), glossary)
    check_translation_invariants(text, translated)
    return translated


def _field_text(case: GoldenTestCase, measure: FieldMeasure) -> str:
    if measure.field == "reference_contexts":
        assert measure.index is not None
        return case.reference_contexts[measure.index]
    return str(getattr(case, measure.field))


def translate_testcase(
    case: GoldenTestCase,
    translator: TextTranslator,
    glossary: Mapping[str, str],
) -> GoldenTestCase:
    """Dịch các trường tiếng Anh của `case`; trả bản sao đã cập nhật (hoặc chính `case` nếu không cần).

    Trường trượt kiểm tra giữ nguyên bản gốc tiếng Anh và gắn cờ `translation_review` (lý do
    đầu tiên); lỗi mạng (`translate_error`) dừng xử lý phần còn lại của mẫu. Dịch
    `reference_contexts` thành công gắn `contexts_translated` nếu không có cờ nặng hơn.

    Raises:
        TranslatorConfigError: Lỗi cấu hình (propagate, không gắn cờ).
    """
    english = detect_english_fields(case)
    if not english:
        return case
    updates: dict[str, Any] = {}
    contexts = list(case.reference_contexts)
    originals: dict[str, str | list[str]] = dict(case.original_en or {})
    review: str | None = None
    contexts_translated = False
    for measure in english:
        source = _field_text(case, measure)
        try:
            translated = translate_field(source, translator, glossary)
        except FieldRejected as rejected:
            review = review or rejected.reason
            continue
        except TranslateError:
            review = REVIEW_TRANSLATE_ERROR
            break
        if measure.field == "reference_contexts":
            assert measure.index is not None
            contexts[measure.index] = translated
            contexts_translated = True
        else:
            updates[measure.field] = translated
            originals[measure.field] = source
    if contexts_translated:
        updates["reference_contexts"] = contexts
        originals["reference_contexts"] = list(case.reference_contexts)
        review = review or REVIEW_CONTEXTS_TRANSLATED
    updates["original_en"] = originals or None
    updates["translation_review"] = review
    return case.model_copy(update=updates)


# ---------------------------------------------------------------------------
# Điều phối trên raw (mục 12.5, 12.6)
# ---------------------------------------------------------------------------

_ROW_FIELDS: Final = (
    "user_input",
    "reference",
    "reference_contexts",
    "original_en",
    "translation_review",
)


class TranslationReport(BaseModel):
    """Kết quả một lần `translate`; chỉ số đếm và số dòng, không nội dung."""

    translated_samples: int = 0
    translated_fields: int = 0
    flagged_this_run: dict[str, int] = Field(default_factory=dict)
    skipped_flagged: int = 0
    not_needed: int = 0
    stopped_by_consecutive_errors: bool = False
    flagged_total: int = 0  # mọi dòng raw còn `translation_review`, kể cả từ lần trước

    @property
    def exit_code(self) -> int:
        """1 dừng do lỗi liên tiếp; 3 xong nhưng còn mẫu cần soát; 0 sạch (mục 12.6)."""
        if self.stopped_by_consecutive_errors:
            return 1
        return 3 if self.flagged_total else 0


def _load_cases(path: Path) -> tuple[list[dict[str, Any]], list[GoldenTestCase]]:
    if not path.exists():
        raise EvalInputError(f"Không thấy {path}; chạy `generate` trước.")
    rows = read_raw_rows(path)
    cases = []
    for position, row in enumerate(rows, start=1):
        try:
            cases.append(GoldenTestCase.model_validate(row))
        except ValidationError as error:
            raise EvalInputError(
                f"{path}: dòng {position} không hợp lệ: {error}"
            ) from error
    return rows, cases


def _select_targets(
    cases: list[GoldenTestCase], *, retry_flagged: bool, limit: int | None
) -> tuple[list[tuple[int, list[FieldMeasure]]], int, int]:
    """Chọn (chỉ số 0-based, trường tiếng Anh) cần dịch; trả thêm số mẫu bị cờ bỏ qua và số không cần."""
    targets: list[tuple[int, list[FieldMeasure]]] = []
    skipped_flagged = not_needed = 0
    for position, case in enumerate(cases):
        english = detect_english_fields(case)
        if not english:
            not_needed += 1
        elif case.translation_review is not None and not retry_flagged:
            skipped_flagged += 1
        elif limit is None or len(targets) < limit:
            targets.append((position, english))
    return targets, skipped_flagged, not_needed


def plan_translation(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    retry_flagged: bool = False,
    limit: int | None = None,
) -> str:
    """Bảng `--dry-run`: trường tiếng Anh theo dòng raw + vùng tỉ lệ mơ hồ; không gọi mạng.

    Raises:
        EvalInputError: Raw thiếu hoặc không hợp lệ.
    """
    _, cases = _load_cases(output_dir / RAW_TESTSET_FILENAME)
    targets, skipped_flagged, _ = _select_targets(
        cases, retry_flagged=retry_flagged, limit=limit
    )
    lines = ["Dòng  Trường                 Tỉ lệ dấu  Ký tự"]
    total_fields = total_chars = 0
    for position, english in targets:
        for measure in english:
            total_fields += 1
            total_chars += measure.chars
            lines.append(
                f"{position + 1:>4}  {measure.label:<22} {measure.ratio:>9.2f}  {measure.chars:>5}"
            )
    lines.append(
        f"Tóm tắt: {len(targets)} mẫu, {total_fields} trường tiếng Anh (tỉ lệ dấu < "
        f"{ENGLISH_MAX_DIACRITIC_RATIO:.2f}), {total_chars} ký tự cần dịch; "
        f"bỏ qua {skipped_flagged} mẫu đã bị cờ (dùng --retry-flagged)."
    )
    low, high = AMBIGUOUS_RATIO_RANGE
    lines.append(f"Vùng mơ hồ (tỉ lệ dấu {low:.2f}-{high:.2f}), cần nhìn tay:")
    for position, case in enumerate(cases):
        for measure in measure_fields(case):
            if low <= measure.ratio <= high:
                verdict = "sẽ dịch" if measure.is_english else "giữ nguyên"
                lines.append(
                    f"{position + 1:>4}  {measure.label:<22} {measure.ratio:>9.2f}  "
                    f"{measure.chars:>5}  ({verdict})"
                )
    return "\n".join(lines)


def _apply_to_row(row: dict[str, Any], case: GoldenTestCase) -> None:
    dumped = case.model_dump()
    for key in _ROW_FIELDS:
        row[key] = dumped[key]


def translate_raw(
    output_dir: Path,
    translator: TextTranslator,
    *,
    limit: int | None = None,
    retry_flagged: bool = False,
) -> TranslationReport:
    """Dịch các mẫu tiếng Anh trong raw tại chỗ, ghi nguyên tử sau MỖI mẫu (mục 12.5, 12.6).

    Chỉ sửa các trường đã dịch và hai cột mới; không sắp xếp lại hay đụng dòng khác.
    Raw tự là checkpoint: trường đã dịch có tỉ lệ dấu cao nên lần chạy sau không chọn lại.

    Raises:
        EvalInputError: Raw/bảng thuật ngữ thiếu hoặc hỏng; `TranslatorConfigError` khi cấu hình
            dịch sai (mã thoát 2). Các mẫu đã dịch trước đó vẫn nằm trong raw.
    """
    path = output_dir / RAW_TESTSET_FILENAME
    rows, cases = _load_cases(path)
    glossary = load_glossary(output_dir / GLOSSARY_FILENAME)
    targets, skipped_flagged, not_needed = _select_targets(
        cases, retry_flagged=retry_flagged, limit=limit
    )
    report = TranslationReport(skipped_flagged=skipped_flagged, not_needed=not_needed)
    consecutive_errors = 0
    for position, _ in targets:
        try:
            translated = translate_testcase(cases[position], translator, glossary)
        except TranslatorConfigError:
            raise
        except Exception as error:  # bug ngoài dự kiến: chỉ ghi vị trí, không nội dung
            logger.error(
                "Dịch dòng %d lỗi không phân loại: %s",
                position + 1,
                _describe_traceback(error),
            )
            raise
        newly_translated = set(translated.original_en or {}) - set(
            cases[position].original_en or {}
        )
        cases[position] = translated
        _apply_to_row(rows[position], translated)
        write_raw_rows(path, rows)
        report.translated_samples += 1
        report.translated_fields += len(newly_translated)
        review = translated.translation_review
        if review is not None:
            report.flagged_this_run[review] = report.flagged_this_run.get(review, 0) + 1
        logger.info("Dịch dòng %d xong, review=%s", position + 1, review)
        consecutive_errors = (
            consecutive_errors + 1 if review == REVIEW_TRANSLATE_ERROR else 0
        )
        if consecutive_errors >= MAX_CONSECUTIVE_TRANSLATE_ERRORS:
            report.stopped_by_consecutive_errors = True
            break
    report.flagged_total = sum(case.translation_review is not None for case in cases)
    return report
