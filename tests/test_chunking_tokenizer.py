"""Test `chunking/tokenizer.py` với tokenizer PhoBERT-based THẬT (mục 6).

Đánh dấu `slow` (nạp model thật từ HuggingFace Hub, ~20s lần đầu) -- CI job
`checks` bỏ qua (`pytest -m "not slow"`), tester chạy riêng để xác nhận
tiêu chí hoàn thành mục 11 (token_count đúng theo tokenizer thật).
"""

from __future__ import annotations

import pytest

from production_legal_qa_rag.chunking.tokenizer import count_tokens

pytestmark = pytest.mark.slow


def test_count_tokens_tra_ve_so_nguyen_duong():
    assert (
        count_tokens(
            "Luật này không áp dụng đối với bảo hiểm y tế mang tính kinh doanh."
        )
        > 0
    )


def test_count_tokens_bao_gom_special_token():
    # add_special_tokens=True (mục 6) -> ngay cả chuỗi rỗng cũng có CLS/SEP.
    assert count_tokens("") >= 2


def test_count_tokens_van_ban_dai_hon_co_nhieu_token_hon():
    short_count = count_tokens("Một câu ngắn.")
    long_count = count_tokens(
        "Luật này quy định về chế độ, chính sách bảo hiểm y tế, bao gồm đối "
        "tượng, mức đóng, trách nhiệm và phương thức đóng bảo hiểm y tế; thẻ "
        "bảo hiểm y tế; phạm vi được hưởng bảo hiểm y tế."
    )
    assert long_count > short_count


def test_count_tokens_deterministic():
    text = "Bảo hiểm y tế là hình thức bảo hiểm bắt buộc."
    assert count_tokens(text) == count_tokens(text)


def test_count_tokens_cache_khong_tinh_lai_cho_cung_1_van_ban():
    text = "Văn bản dùng để kiểm tra cache lru_cache của count_tokens."
    count_tokens.cache_clear()
    count_tokens(text)
    hits_before = count_tokens.cache_info().hits
    count_tokens(text)
    hits_after = count_tokens.cache_info().hits
    assert hits_after == hits_before + 1


def test_count_tokens_word_segment_anh_huong_ket_qua():
    # Ghép từ ("bảo_hiểm_y_tế") do pyvi word-segment sinh ra -- PhoBERT có thể
    # coi cụm từ ghép là 1 token khác so với đứng riêng lẻ, nhưng quan trọng
    # nhất là: kết quả không rỗng/không lỗi khi văn bản có nhiều cụm từ ghép
    # tiếng Việt liên tiếp.
    assert count_tokens("Bảo hiểm y tế toàn dân là việc các đối tượng tham gia.") > 0
