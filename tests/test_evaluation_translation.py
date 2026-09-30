"""Test cho `evaluation/translation.py`: hàm thuần và client Apps Script (evaluation_spec.md mục 12.2-12.4).

Gồm: phát hiện trường tiếng Anh (12.3), tách/chuẩn hoá/bảng thuật ngữ/kiểm bất biến (12.4),
`AppsScriptTranslator` với HTTP giả (`get`/`sleep`/`jitter` tiêm được), `TranslateSettings`
và `translate_field`. Không cần `ragas`, không gọi mạng thật, không đọc `.env` (mọi test nạp
cấu hình từ môi trường đều đổi cwd sang thư mục tạm).
"""

from __future__ import annotations

import json
import random
import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from production_legal_qa_rag.evaluation import translation as tr
from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.testset_generator import EvalInputError

URL = "https://script.example.invalid/macros/s/SECRET-DEPLOY-ID/exec"
KEY = "s3cr3t-key-value"

EN_Q1 = "What must the employer do under Article 5 of the Labor Code?"
VI_Q1 = "Người sử dụng lao động phải làm gì theo Điều 5 của Bộ luật Lao động?"
EN_A1 = "The employer must pay wages on time as stated in Clause 2 of Article 5."
VI_A1 = "Người sử dụng lao động phải trả lương đúng hạn theo Khoản 2 Điều 5."
VI_CONTEXT = (
    "Điều 5. Quyền và nghĩa vụ của người sử dụng lao động\n"
    "1. Người sử dụng lao động có quyền tuyển dụng, bố trí, quản lý lao động."
)
EN_CONTEXT = (
    "Article 5. Rights and obligations of the employer\n"
    "1. The employer has the right to recruit, assign and manage employees."
)
GLOSSARY = {"lương hưu xã hội": "trợ cấp hưu trí xã hội"}


def _settings(key: str | None = None, timeout: int = 60) -> tr.TranslateSettings:
    # `model_construct` tránh đọc `.env`/môi trường thật.
    return tr.TranslateSettings.model_construct(
        url=URL, key=key, timeout_seconds=timeout
    )


def _case(
    user_input: str = VI_Q1,
    reference: str = VI_A1,
    contexts: list[str] | None = None,
) -> GoldenTestCase:
    return GoldenTestCase.model_validate(
        {
            "user_input": user_input,
            "reference": reference,
            "reference_contexts": contexts if contexts is not None else [VI_CONTEXT],
        }
    )


@dataclass(frozen=True)
class FakeResponse:
    status_code: int
    content: bytes


def _ok(text: str) -> FakeResponse:
    return FakeResponse(200, json.dumps({"translatedText": text}).encode("utf-8"))


def _json(payload: object, status: int = 200) -> FakeResponse:
    return FakeResponse(status, json.dumps(payload).encode("utf-8"))


class ScriptedGet:
    """`get` giả: trả/ném lần lượt các kết quả đã hẹn; kết quả cuối lặp lại vô hạn."""

    def __init__(self, *outcomes: FakeResponse | BaseException) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(
        self, url: str, params: Mapping[str, str], timeout: float
    ) -> FakeResponse:
        self.calls.append((url, dict(params), timeout))
        if len(self._outcomes) > 1:
            outcome = self._outcomes.pop(0)
        else:
            outcome = self._outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class Sleeps:
    def __init__(self) -> None:
        self.values: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.values.append(seconds)


def _translator(
    get: ScriptedGet, sleeps: Sleeps | None = None, key: str | None = None
) -> tr.AppsScriptTranslator:
    return tr.AppsScriptTranslator(
        _settings(key=key),
        get=get,
        sleep=sleeps if sleeps is not None else Sleeps(),
        jitter=lambda: 0.25,
    )


class FakeTranslator:
    """`TextTranslator` giả: tra từ điển theo đúng đoạn gửi đi; giá trị `Exception` thì ném."""

    def __init__(self, responses: Mapping[str, str | Exception]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        response = self._responses[text]
        if isinstance(response, Exception):
            raise response
        return response


# ---------------------------------------------------------------------------
# 12.3 Phát hiện trường tiếng Anh
# ---------------------------------------------------------------------------


def test_diacritic_ratio_none_khi_duoi_bon_token_chu():
    assert tr.diacritic_ratio("Xin chào bạn") is None
    assert tr.diacritic_ratio("") is None
    # Token không có chữ cái ("1.", "—") không được tính vào mẫu số.
    assert tr.diacritic_ratio("1. 2. 3. — Xin chào bạn") is None


def test_diacritic_ratio_tieng_anh_la_khong_tieng_viet_gan_mot():
    english = tr.diacritic_ratio("The employer must pay wages on time")
    vietnamese = tr.diacritic_ratio("Người sử dụng lao động phải trả lương đúng hạn")

    assert english == 0.0
    assert vietnamese == pytest.approx(9 / 10)  # chỉ "lao" không có dấu


def test_diacritic_ratio_dem_chu_d_gach_va_dau_to_hop_roi():
    assert tr.diacritic_ratio("đ Đ đ Đ") == 1.0
    composed = unicodedata.normalize("NFC", "Người lao động được nghỉ")
    decomposed = unicodedata.normalize("NFD", composed)
    assert tr.diacritic_ratio(decomposed) == tr.diacritic_ratio(composed)
    assert tr.diacritic_ratio(composed) == pytest.approx(4 / 5)


def test_diacritic_ratio_chu_hon_hop_voi_tieng_anh():
    # Người, động, có, quyền có dấu; lao, work, rest không -> 4/7.
    ratio = tr.diacritic_ratio("Người lao động có quyền work rest")

    assert ratio == pytest.approx(4 / 7)


def test_is_english_nguong_la_nho_hon_tuyet_doi_0_10():
    assert tr.ENGLISH_MAX_DIACRITIC_RATIO == 0.10
    at_threshold = " ".join(["xã"] + ["abc"] * 9)  # 1/10 = 0,10: KHÔNG phải tiếng Anh
    below = " ".join(["xã"] + ["abc"] * 19)  # 1/20 = 0,05: tiếng Anh
    assert not tr.is_english(at_threshold)
    assert tr.is_english(below)
    assert tr.is_english(" ".join(["abc"] * 10))


def test_is_english_false_khi_qua_ngan_de_phan_loai():
    assert not tr.is_english("Hello there friend")  # 3 token chữ
    assert not tr.is_english("")


def test_detect_english_fields_doc_lap_tung_truong():
    only_reference = _case(user_input=VI_Q1, reference=EN_A1)
    only_user_input = _case(user_input=EN_Q1, reference=VI_A1)
    both = _case(user_input=EN_Q1, reference=EN_A1)
    neither = _case()

    assert [m.field for m in tr.detect_english_fields(only_reference)] == ["reference"]
    found = tr.detect_english_fields(only_user_input)
    assert [m.field for m in found] == ["user_input"]
    found = tr.detect_english_fields(both)
    assert [m.field for m in found] == ["user_input", "reference"]
    assert tr.detect_english_fields(neither) == []


def test_detect_english_fields_reference_contexts_theo_chi_so_va_bo_truong_ngan():
    case = _case(contexts=[VI_CONTEXT, EN_CONTEXT, "Short one"])

    measures = tr.detect_english_fields(case)

    assert [(m.field, m.index) for m in measures] == [("reference_contexts", 1)]
    assert measures[0].label == "reference_contexts[1]"
    assert measures[0].chars == len(EN_CONTEXT)
    assert measures[0].ratio == 0.0


def test_measure_fields_bo_truong_ngan_va_khong_danh_dau_tieng_viet():
    case = _case(user_input="Có không?", reference=VI_A1, contexts=[])

    measures = tr.measure_fields(case)

    assert [m.label for m in measures] == ["reference"]
    assert not measures[0].is_english


# ---------------------------------------------------------------------------
# 12.4 Tách theo dòng
# ---------------------------------------------------------------------------


def test_split_for_request_van_ban_ngan_la_mot_phan():
    assert tr.split_for_request("one line") == ["one line"]
    assert tr.split_for_request("") == [""]
    assert tr.MAX_REQUEST_CHARS == 4_000


def test_split_for_request_bien_chinh_xac_max_chars():
    # 4 + 1 + 2 = 7 ký tự: vừa khít giới hạn 7 -> một phần; giới hạn 6 thì tách.
    assert tr.split_for_request("aaaa\nbb", max_chars=7) == ["aaaa\nbb"]
    assert tr.split_for_request("aaaa\nbb", max_chars=6) == ["aaaa", "bb"]


def test_split_for_request_noi_lai_bang_xuong_dong_ra_dung_ban_goc():
    text = "dòng một\n\ndòng ba dài hơn\nd4\n"

    parts = tr.split_for_request(text, max_chars=20)

    assert len(parts) > 1
    assert "\n".join(parts) == text
    assert all(len(part) <= 20 for part in parts)


def test_split_for_request_dong_qua_dai_la_too_long():
    text = "ngắn\n" + "x" * 11 + "\nngắn"

    with pytest.raises(tr.FieldRejected) as caught:
        tr.split_for_request(text, max_chars=10)

    assert caught.value.reason == tr.REVIEW_TOO_LONG == "too_long"


def test_split_for_request_dong_dung_bang_gioi_han_khong_bi_tu_choi():
    assert tr.split_for_request("x" * 10, max_chars=10) == ["x" * 10]


def test_split_for_request_thuoc_tinh_ngau_nhien_seed_co_dinh():
    rng = random.Random(20260930)
    for _ in range(200):
        max_chars = rng.randint(10, 40)
        lines = ["x" * rng.randint(0, max_chars) for _ in range(rng.randint(1, 30))]
        text = "\n".join(lines)

        parts = tr.split_for_request(text, max_chars=max_chars)

        assert "\n".join(parts) == text
        assert all(len(part) <= max_chars for part in parts)
        # Gộp tham lam: không thể nhét dòng đầu của phần sau vào phần trước.
        for previous, following in pairwise(parts):
            first_line = following.split("\n")[0]
            assert len(previous) + 1 + len(first_line) > max_chars


# ---------------------------------------------------------------------------
# 12.4 Chuẩn hoá nhẹ
# ---------------------------------------------------------------------------


def test_normalize_vietnamese_nfc_khoang_trang_va_dau_cau():
    decomposed = unicodedata.normalize("NFD", "Người  lao\tđộng  , được  nghỉ .")

    assert tr.normalize_vietnamese(decomposed) == "Người lao động, được nghỉ."


def test_normalize_vietnamese_tach_chu_hoa_dinh_chu_thuong():
    glued = tr.normalize_vietnamese("quyền củaNgười lao động")

    assert glued == "quyền của Người lao động"
    assert tr.normalize_vietnamese("nhàĐất") == "nhà Đất"


def test_normalize_vietnamese_khong_sua_dinh_chu_thuong_voi_thuong():
    assert tr.normalize_vietnamese("bảo vệsức khỏe") == "bảo vệsức khỏe"
    assert tr.normalize_vietnamese("Ngườilao động") == "Ngườilao động"


def test_normalize_vietnamese_giu_xuong_dong_va_cat_le_moi_dong():
    text = "  dòng một  \n\n   dòng hai ,\n"

    assert tr.normalize_vietnamese(text) == "dòng một\n\ndòng hai,\n"


def test_normalize_vietnamese_luy_dang():
    text = unicodedata.normalize("NFD", "quyềnNgười  ,  lao  động .")

    once = tr.normalize_vietnamese(text)

    assert tr.normalize_vietnamese(once) == once


# ---------------------------------------------------------------------------
# 12.4 Bảng thuật ngữ
# ---------------------------------------------------------------------------


def test_apply_glossary_thay_khong_phan_biet_hoa_thuong_va_giu_hoa_dau():
    lower = tr.apply_glossary("Được hưởng lương hưu xã hội hằng tháng.", GLOSSARY)
    capital = tr.apply_glossary("Lương hưu xã hội là gì?", GLOSSARY)
    upper = tr.apply_glossary("LƯƠNG HƯU XÃ HỘI", GLOSSARY)

    assert lower == "Được hưởng trợ cấp hưu trí xã hội hằng tháng."
    assert capital == "Trợ cấp hưu trí xã hội là gì?"
    assert upper == "Trợ cấp hưu trí xã hội"


def test_apply_glossary_cum_dai_truoc():
    glossary = {"hưu xã hội": "SHORT", "lương hưu xã hội": "LONG"}

    result = tr.apply_glossary("lương hưu xã hội và hưu xã hội", glossary)

    assert result == "LONG và SHORT"


def test_apply_glossary_rong_hoac_khoa_trong_thi_khong_doi():
    text = "Người lao động có quyền nghỉ."

    assert tr.apply_glossary(text, {}) == text
    assert tr.apply_glossary(text, {"": "X", "   ": "Y"}) == text


def test_apply_glossary_khoa_la_chuoi_chu_khong_phai_regex():
    assert tr.apply_glossary("axb và a.b", {"a.b": "ĐÚNG"}) == "axb và ĐÚNG"


def test_apply_glossary_khop_khoa_nfd_trong_bang():
    glossary = {unicodedata.normalize("NFD", "lương hưu xã hội"): "trợ cấp"}

    assert tr.apply_glossary("lương hưu xã hội", glossary) == "trợ cấp"


def test_load_glossary_thieu_file_la_bang_rong(tmp_path: Path):
    assert tr.load_glossary(tmp_path / "translation_glossary.json") == {}
    assert tr.GLOSSARY_FILENAME == "translation_glossary.json"


def test_load_glossary_doc_duoc_json_utf8(tmp_path: Path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps(GLOSSARY, ensure_ascii=False), encoding="utf-8")

    assert tr.load_glossary(path) == GLOSSARY


@pytest.mark.parametrize(
    "content",
    ["{hỏng", '["a", "b"]', '{"a": 1}', '{"a": null}', "42"],
)
def test_load_glossary_sai_dinh_dang_la_loi_dau_vao(tmp_path: Path, content: str):
    path = tmp_path / "g.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(EvalInputError):
        tr.load_glossary(path)


def test_load_glossary_khong_phai_utf8_la_loi_dau_vao(tmp_path: Path):
    path = tmp_path / "g.json"
    path.write_bytes("{}".encode("utf-16"))

    with pytest.raises(EvalInputError):
        tr.load_glossary(path)


# ---------------------------------------------------------------------------
# 12.4 Tham chiếu pháp lý
# ---------------------------------------------------------------------------


def test_extract_legal_references_tieng_anh_anh_xa_sang_loai_tieng_viet():
    text = "Article 5, Clause 2, Point a of Chapter IV, Section 3, Paragraph 7."

    found = tr.extract_legal_references(text, language="en")

    assert found == {
        ("điều", "5"),
        ("khoản", "2"),
        ("điểm", "a"),
        ("chương", "iv"),
        ("mục", "3"),
        ("khoản", "7"),  # Paragraph -> Khoản
    }


def test_extract_legal_references_tieng_viet():
    text = "Điều 5, Khoản 2, Điểm a, Chương IV, Mục 3."

    found = tr.extract_legal_references(text, language="vi")

    assert found == {
        ("điều", "5"),
        ("khoản", "2"),
        ("điểm", "a"),
        ("chương", "iv"),
        ("mục", "3"),
    }


def test_extract_legal_references_khong_phan_biet_hoa_thuong_va_dang_so_nhieu():
    english = tr.extract_legal_references("ARTICLE 5 and articles 6", language="en")
    vietnamese = tr.extract_legal_references("điều 5; ĐIỂM A; điểm a", language="vi")

    assert english == {("điều", "5"), ("điều", "6")}
    assert vietnamese == {("điều", "5"), ("điểm", "a")}


def test_extract_legal_references_so_sanh_tap_khong_dem_so_lan():
    once = tr.extract_legal_references("Article 5", language="en")
    many = tr.extract_legal_references("Article 5, see Article 5", language="en")

    assert once == many == {("điều", "5")}


def test_extract_legal_references_diem_chu_d_gach_ngang():
    found = tr.extract_legal_references("Điểm đ khoản 1 Điều 3", language="vi")

    assert found == {("điểm", "đ"), ("khoản", "1"), ("điều", "3")}


def test_extract_legal_references_khong_nham_muc_dich_hay_point_of():
    assert tr.extract_legal_references("Mục đích của luật", language="vi") == set()
    assert tr.extract_legal_references("point of fact", language="en") == set()


def test_extract_legal_references_khong_co_cum_nao():
    assert tr.extract_legal_references("No references", language="en") == set()
    assert tr.extract_legal_references("Không có gì cả", language="vi") == set()


def test_extract_legal_references_chuan_hoa_nfc_truoc_khi_so():
    decomposed = unicodedata.normalize("NFD", "Điều 5 và Điểm đ")

    found = tr.extract_legal_references(decomposed, language="vi")

    assert found == {("điều", "5"), ("điểm", "đ")}


# ---------------------------------------------------------------------------
# 12.4 Kiểm bất biến
# ---------------------------------------------------------------------------


def _reason(original: str, translated: str) -> str | None:
    try:
        tr.check_translation_invariants(original, translated)
    except tr.FieldRejected as rejected:
        return rejected.reason
    return None


def test_invariants_ban_dich_dung_thi_khong_loi():
    assert _reason(EN_Q1, VI_Q1) is None
    assert _reason(EN_A1, VI_A1) is None


def test_invariants_rong_hoac_van_tieng_anh_la_still_english():
    assert _reason(EN_Q1, "") == "still_english"
    assert _reason(EN_Q1, "   \n ") == "still_english"
    assert _reason(EN_Q1, EN_Q1) == "still_english"


def test_invariants_ban_dich_qua_ngan_khong_bi_coi_la_still_english():
    assert _reason("Yes, exactly.", "Đúng.") is None


def test_invariants_lech_so_dieu_la_citation_mismatch():
    assert _reason(EN_Q1, VI_Q1.replace("Điều 5", "Điều 6")) == "citation_mismatch"


def test_invariants_mat_hoac_them_tham_chieu_la_citation_mismatch():
    lost = "Người sử dụng lao động phải làm gì theo Bộ luật Lao động?"
    extra = VI_Q1.replace("Điều 5", "Điều 5 Khoản 1")

    assert _reason(EN_Q1, lost) == "citation_mismatch"
    assert _reason(EN_Q1, extra) == "citation_mismatch"


def test_invariants_goc_khong_co_tham_chieu_thi_khong_kiem_muc_nay():
    original = "What are the general duties of every employer in the country?"
    translated = "Nghĩa vụ chung của mọi người sử dụng lao động theo Điều 9 là gì?"

    assert _reason(original, translated) is None


def test_invariants_loai_tham_chieu_khac_nhau_van_lech():
    assert _reason(EN_Q1, VI_Q1.replace("Điều 5", "Khoản 5")) == "citation_mismatch"


def test_invariants_paragraph_tuong_duong_khoan_va_so_tap_khong_so_lan():
    original = "Paragraph 2 sets the wage. Paragraph 2 also sets the leave days."
    translated = "Khoản 2 quy định tiền lương. Nó cũng quy định số ngày nghỉ phép."

    assert _reason(original, translated) is None


def test_invariants_diem_khong_phan_biet_hoa_thuong():
    original = "Under Point a the employer must pay wages on time and in full."
    translated = "Theo Điểm A người sử dụng lao động phải trả lương đúng hạn đầy đủ."

    assert _reason(original, translated) is None


def test_invariants_lech_so_dong_khong_rong_la_line_count_mismatch():
    original = "First line about wages owed.\nSecond line about leave.\nThird one."
    merged = "Dòng một về tiền lương còn nợ.\nDòng hai về ngày nghỉ phép và dòng ba."

    assert _reason(original, merged) == "line_count_mismatch"


def test_invariants_dong_trong_khong_tinh_vao_so_dong():
    original = "First line about wages owed.\n\nSecond line about leave days."
    translated = "Dòng một về tiền lương còn nợ.\nDòng hai về ngày nghỉ phép."

    assert _reason(original, translated) is None


def test_invariants_thu_tu_kiem_still_english_roi_citation_roi_dong():
    original = "Article 5 lists duties.\nThe employer must act."
    both_wrong = "Điều 7 quy định nghĩa vụ và người sử dụng lao động phải làm."
    untranslated = "Article 5 lists duties. The employer must act."

    assert _reason(original, both_wrong) == "citation_mismatch"
    assert _reason(original, untranslated) == "still_english"


# ---------------------------------------------------------------------------
# TranslateSettings
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """Đổi cwd sang thư mục tạm (không đọc `.env` thật) và xoá biến TRANSLATE_*."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRANSLATE_URL", raising=False)
    monkeypatch.delenv("TRANSLATE_KEY", raising=False)
    return monkeypatch


def test_load_translate_settings_doc_url_key_tu_moi_truong(
    clean_env: pytest.MonkeyPatch,
):
    clean_env.setenv("TRANSLATE_URL", URL)
    clean_env.setenv("TRANSLATE_KEY", KEY)

    settings = tr.load_translate_settings()

    assert settings.url == URL
    assert settings.key == KEY
    assert settings.timeout_seconds == 60


def test_load_translate_settings_key_tuy_chon(clean_env: pytest.MonkeyPatch):
    clean_env.setenv("TRANSLATE_URL", URL)

    assert tr.load_translate_settings().key is None


def test_load_translate_settings_key_de_trong_coi_nhu_chua_dat(
    clean_env: pytest.MonkeyPatch,
):
    clean_env.setenv("TRANSLATE_URL", URL)
    clean_env.setenv("TRANSLATE_KEY", "")

    assert tr.load_translate_settings().key is None


@pytest.mark.parametrize("value", [None, ""])
def test_load_translate_settings_thieu_url_la_loi_dau_vao_nen_ten_bien(
    clean_env: pytest.MonkeyPatch, value: str | None
):
    if value is not None:
        clean_env.setenv("TRANSLATE_URL", value)
    clean_env.setenv("TRANSLATE_KEY", KEY)

    with pytest.raises(EvalInputError, match="TRANSLATE_URL") as caught:
        tr.load_translate_settings()

    assert KEY not in str(caught.value)


def test_load_translate_settings_bo_qua_bien_thua_trong_env_file(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
):
    env_text = f"TRANSLATE_URL={URL}\nUNRELATED_SETTING=1\n"
    (tmp_path / ".env").write_text(env_text, encoding="utf-8")

    assert tr.load_translate_settings().url == URL


def test_translate_settings_url_khong_duoc_rong():
    with pytest.raises(ValidationError):
        tr.TranslateSettings.model_validate({"TRANSLATE_URL": ""})


# ---------------------------------------------------------------------------
# AppsScriptTranslator (12.2, 12.4 bước 2)
# ---------------------------------------------------------------------------


def test_translator_goi_get_dung_tham_so_va_tra_ban_dich():
    get = ScriptedGet(_ok("Xin chào"))

    result = _translator(get).translate("Hello there")

    assert result == "Xin chào"
    [(url, params, timeout)] = get.calls
    assert url == URL
    assert params == {"text": "Hello there", "source": "en", "target": "vi"}
    assert timeout == 60


def test_translator_gui_key_khi_co_va_khong_gui_khi_khong_co():
    with_key, without_key = ScriptedGet(_ok("a b")), ScriptedGet(_ok("a b"))

    _translator(with_key, key=KEY).translate("x")
    _translator(without_key, key=None).translate("x")

    assert with_key.calls[0][1]["key"] == KEY
    assert "key" not in without_key.calls[0][1]


def test_translator_khong_retry_khi_thanh_cong():
    sleeps = Sleeps()
    get = ScriptedGet(_ok("Xin chào"))

    _translator(get, sleeps).translate("Hello")

    assert len(get.calls) == 1
    assert sleeps.values == []


@pytest.mark.parametrize("status", [500, 502, 503, 429])
def test_translator_retry_5xx_va_429_roi_thanh_cong_voi_backoff_jitter(status: int):
    sleeps = Sleeps()
    failing = FakeResponse(status, b"")
    get = ScriptedGet(failing, failing, _ok("Được"))

    assert _translator(get, sleeps).translate("Ok") == "Được"

    assert len(get.calls) == 3
    assert sleeps.values == [1.25, 2.25]  # 1 * 2**0 + jitter, 1 * 2**1 + jitter


def test_translator_het_retry_la_translate_error_khong_ngu_them_lan_cuoi():
    sleeps = Sleeps()
    get = ScriptedGet(FakeResponse(503, b"down"))

    with pytest.raises(tr.TranslateError):
        _translator(get, sleeps).translate("Hello")

    assert len(get.calls) == tr.MAX_ATTEMPTS == 3
    assert len(sleeps.values) == 2


@pytest.mark.parametrize(
    "error",
    [ConnectionError(), TimeoutError(), OSError()],
    ids=["connection", "timeout", "oserror"],
)
def test_translator_retry_loi_mang_roi_thanh_cong(error: OSError):
    get = ScriptedGet(error, _ok("Xong"))

    assert _translator(get).translate("Done") == "Xong"
    assert len(get.calls) == 2


def test_translator_thong_diep_loi_khong_ro_ri_url_key_hay_noi_dung():
    leaking = ConnectionError(f"pool for {URL}?key={KEY}&text=SECRET-TEXT")
    get = ScriptedGet(leaking)

    with pytest.raises(tr.TranslateError) as caught:
        _translator(get, key=KEY).translate("SECRET-TEXT")

    message = str(caught.value)
    assert "ConnectionError" in message
    for secret in (URL, KEY, "SECRET-TEXT", "SECRET-DEPLOY-ID"):
        assert secret not in message
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(200, b"not json at all"),
        FakeResponse(200, b""),
        FakeResponse(200, b"\xff\xfe\x00"),
        _json(["translatedText"]),
        _json("translatedText"),
        _json({}),
        _json({"translatedText": ""}),
        _json({"translatedText": "   "}),
        _json({"translatedText": 42}),
        _json({"translatedText": None}),
        _json({"error": "quota exceeded"}),
        _json({"error": {"code": 1}}),
    ],
    ids=[
        "khong_phai_json",
        "rong",
        "khong_utf8",
        "json_list",
        "json_chuoi",
        "thieu_truong",
        "translatedText_rong",
        "translatedText_khoang_trang",
        "translatedText_khong_phai_chuoi",
        "translatedText_null",
        "error_khac",
        "error_doi_tuong",
    ],
)
def test_translator_phan_hoi_khong_hop_le_duoc_retry_roi_translate_error(
    response: FakeResponse,
):
    get = ScriptedGet(response)

    with pytest.raises(tr.TranslateError):
        _translator(get).translate("Hello")

    assert len(get.calls) == 3


def test_translator_phan_hoi_hong_tam_thoi_roi_hoi_phuc():
    get = ScriptedGet(FakeResponse(200, b"oops"), _ok("Ổn rồi"))

    assert _translator(get).translate("Fine") == "Ổn rồi"


@pytest.mark.parametrize(
    "body",
    [
        b"<!DOCTYPE html><html><body>Sign in</body></html>",
        b"  \n<html><title>Error</title></html>",
    ],
)
def test_translator_html_la_loi_cau_hinh_khong_retry(body: bytes):
    sleeps = Sleeps()
    get = ScriptedGet(FakeResponse(200, body))

    with pytest.raises(tr.TranslatorConfigError):
        _translator(get, sleeps).translate("Hello")

    assert len(get.calls) == 1
    assert sleeps.values == []


@pytest.mark.parametrize("error", ["forbidden", "Forbidden", "  FORBIDDEN "])
def test_translator_forbidden_la_loi_cau_hinh_khong_retry_va_khong_lo_key(error: str):
    sleeps = Sleeps()
    get = ScriptedGet(_json({"error": error}))

    with pytest.raises(tr.TranslatorConfigError) as caught:
        _translator(get, sleeps, key=KEY).translate("Hello")

    assert len(get.calls) == 1
    assert sleeps.values == []
    assert KEY not in str(caught.value)
    assert URL not in str(caught.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_translator_4xx_tru_429_la_loi_cau_hinh_khong_retry(status: int):
    get = ScriptedGet(FakeResponse(status, b'{"translatedText": "x y"}'))

    with pytest.raises(tr.TranslatorConfigError):
        _translator(get).translate("Hello")

    assert len(get.calls) == 1


def test_translator_config_error_la_eval_input_error_de_cli_thoat_ma_2():
    assert issubclass(tr.TranslatorConfigError, EvalInputError)
    assert not issubclass(tr.TranslateError, EvalInputError)


def test_translator_mac_dinh_dung_requests_get_voi_params_va_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[tuple[str, dict[str, str], float]] = []

    def fake_get(url: str, *, params: dict[str, str], timeout: float) -> FakeResponse:
        seen.append((url, params, timeout))
        return _ok("Chào")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get))

    result = tr.AppsScriptTranslator(_settings(key=KEY, timeout=7)).translate("Hi")

    assert result == "Chào"
    expected_params = {"text": "Hi", "source": "en", "target": "vi", "key": KEY}
    assert seen == [(URL, expected_params, 7)]


def test_translator_thieu_thu_vien_requests_la_loi_cau_hinh(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "requests", None)

    with pytest.raises(tr.TranslatorConfigError, match="requests"):
        tr.AppsScriptTranslator(_settings()).translate("Hi")


# ---------------------------------------------------------------------------
# translate_field
# ---------------------------------------------------------------------------


def test_translate_field_tach_dich_chuan_hoa_thuat_ngu_va_kiem_bat_bien():
    en = "Article 5 covers the social pension paid each month."
    raw_vi = "Điều 5  quy định Lương hưu xã hội được chi trả hằng tháng ."
    translator = FakeTranslator({en: raw_vi})

    result = tr.translate_field(en, translator, GLOSSARY)

    expected = "Điều 5 quy định Trợ cấp hưu trí xã hội được chi trả hằng tháng."
    assert translator.calls == [en]
    assert result == expected


def test_translate_field_dai_hon_gioi_han_dich_tung_phan_va_noi_bang_xuong_dong():
    line_1 = " ".join(["The employer must pay wages on time."] * 70)
    line_2 = " ".join(["The employee must follow lawful orders."] * 65)
    line_3 = "Article 5 applies to every contract."
    vi_1 = " ".join(["Người sử dụng lao động phải trả lương đúng hạn."] * 60)
    vi_2 = " ".join(["Người lao động phải tuân theo chỉ thị hợp pháp."] * 55)
    vi_3 = "Điều 5 áp dụng cho mọi hợp đồng."
    assert len(line_1) < tr.MAX_REQUEST_CHARS and len(line_2) < tr.MAX_REQUEST_CHARS
    assert len(line_1) + len(line_2) > tr.MAX_REQUEST_CHARS
    part_2 = f"{line_2}\n{line_3}"
    translator = FakeTranslator(
        {line_1: vi_1, part_2: f"{vi_2}\n{vi_3}"},
    )

    result = tr.translate_field(f"{line_1}\n{part_2}", translator, {})

    assert translator.calls == [line_1, part_2]
    assert result == f"{vi_1}\n{vi_2}\n{vi_3}"


def test_translate_field_dong_qua_dai_bi_tu_choi_truoc_khi_goi_mang():
    text = "word " * 1000  # một dòng 5.000 ký tự > MAX_REQUEST_CHARS
    translator = FakeTranslator({})

    with pytest.raises(tr.FieldRejected) as caught:
        tr.translate_field(text, translator, {})

    assert caught.value.reason == "too_long"
    assert translator.calls == []


def test_translate_field_ban_dich_truot_kiem_bat_bien_nem_field_rejected():
    translator = FakeTranslator({EN_Q1: VI_Q1.replace("Điều 5", "Điều 6")})

    with pytest.raises(tr.FieldRejected) as caught:
        tr.translate_field(EN_Q1, translator, {})

    assert caught.value.reason == "citation_mismatch"


def test_translate_field_loi_mang_va_loi_cau_hinh_duoc_truyen_nguyen():
    network = FakeTranslator({EN_Q1: tr.TranslateError("x")})
    config = FakeTranslator({EN_Q1: tr.TranslatorConfigError("x")})

    with pytest.raises(tr.TranslateError):
        tr.translate_field(EN_Q1, network, {})
    with pytest.raises(tr.TranslatorConfigError):
        tr.translate_field(EN_Q1, config, {})


def test_translate_field_glossary_lam_lech_tham_chieu_van_bi_bat():
    en = "Article 5 covers the social pension paid each month to every person."
    vi = "Điều 5 quy định lương hưu xã hội được chi trả hằng tháng cho mỗi người."
    translator = FakeTranslator({en: vi})

    with pytest.raises(tr.FieldRejected) as caught:
        tr.translate_field(en, translator, {"Điều 5": "Điều 9"})

    assert caught.value.reason == "citation_mismatch"
