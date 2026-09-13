"""Bộ test cho bước chunking (Task 16 của ``chunker_spec.md``).

Chạy trên corpus thật trong ``data/markdown``, không cần PostgreSQL. Chunk toàn
bộ corpus một lần duy nhất ở fixture phạm vi session vì nạp tokenizer và đếm
token là phần đắt nhất.
"""

from __future__ import annotations

import json
import re

from pathlib import Path

import pytest

# ``pre_chunker/`` đã bị xoá; nguồn duy nhất còn lại là bản gộp ``chunking.py``,
# nơi mọi tên của 18 module cũ nằm chung một không gian tên.
from rag.ingestion.chunking import (
    BREADCRUMB_HARD_CAP,
    BUDGET,
    LEVEL_NONE,
    MAX_TOKENS,
    Chunk,
    all_cells,
    build_tree,
    check_model_max_length,
    chunk_markdown,
    count_tokens,
    errors,
    find_negations,
    mark_deindexed,
    parse_blocks,
    parse_table,
    project_root,
    split_sentences,
    to_plain,
    truncate_to_tokens,
    validate,
)


MD_DIR = project_root() / "data" / "markdown"

TNCN = "Luật thuế thu nhập cá nhân"
BHXH = "Luật bảo hiểm xã hội"
ND293 = "Quy định mức lương tối thiểu"

RE_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return RE_WS.sub(" ", text).strip()


@pytest.fixture(scope="session")
def corpus() -> dict[str, list[Chunk]]:
    """Toàn bộ corpus đã chunk và đã đánh dấu ``is_indexed``."""
    documents = {
        path.stem: chunk_markdown(path.read_text(encoding="utf-8"))
        for path in sorted(MD_DIR.glob("*.md"))
    }
    mark_deindexed([chunk for chunks in documents.values() for chunk in chunks])
    return documents


@pytest.fixture(scope="session")
def every_chunk(corpus) -> list[Chunk]:
    return [chunk for chunks in corpus.values() for chunk in chunks]


def pick(chunks: list[Chunk], **fields) -> list[Chunk]:
    return [
        chunk
        for chunk in chunks
        if all(getattr(chunk, name) == value for name, value in fields.items())
    ]


# ==========================================================================
# TASK 1 — đếm token
# ==========================================================================

def test_model_max_length_dung_192():
    assert check_model_max_length() == MAX_TOKENS


def test_chuoi_rong_van_co_hai_token_dac_biet():
    assert count_tokens("") == 2


@pytest.mark.parametrize("limit", [12, 30, 48, 150])
def test_truncate_khong_bao_gio_vuot_tran(limit, every_chunk):
    for chunk in every_chunk[:200]:
        assert count_tokens(truncate_to_tokens(chunk.content, limit)) <= limit


# ==========================================================================
# TASK 2, 3 — parser và cây tài liệu
# ==========================================================================

def test_block_parser_tren_nd293():
    blocks = parse_blocks((MD_DIR / f"{ND293}.md").read_text(encoding="utf-8"))
    assert sum(1 for block in blocks if block.kind == "frontmatter") == 1
    assert sum(1 for block in blocks if block.kind == "heading" and block.level == 1) == 1
    assert sum(1 for block in blocks if block.kind == "heading" and block.level == 4) == 5
    assert sum(1 for block in blocks if block.kind == "heading" and block.level == 5) == 48

    tables = [block for block in blocks if block.kind == "table"]
    assert len(tables) == 1, "bảng Điều 3 phải là MỘT block"
    assert tables[0].line_end - tables[0].line_start + 1 == 6


def test_so_nut_cua_cay_khop_so_lieu_corpus():
    khoan = dieu = dieu_khong_khoan = 0
    for path in sorted(MD_DIR.glob("*.md")):
        for node in build_tree(parse_blocks(path.read_text(encoding="utf-8"))).walk():
            if node.kind in ("khoan", "phu_luc_muc"):
                khoan += 1
            elif node.kind == "dieu":
                dieu += 1
                if not node.children:
                    dieu_khong_khoan += 1
    assert (khoan, dieu, dieu_khong_khoan) == (1957, 567, 72)


# ==========================================================================
# TASK 13 — validator phải xanh trên toàn corpus
# ==========================================================================

def test_khong_con_loi_muc_error(corpus):
    for stem, chunks in corpus.items():
        found = errors(validate(chunks, (MD_DIR / f"{stem}.md").read_text(encoding="utf-8")))
        assert not found, f"{stem}: {[str(issue) for issue in found[:5]]}"


# ==========================================================================
# TASK 16 — bất biến trên toàn corpus
# ==========================================================================

def test_bat_bien_toan_corpus(every_chunk):
    for chunk in every_chunk:
        assert chunk.token_count <= BUDGET
        assert count_tokens(chunk.embedding_text) <= BUDGET
        assert chunk.content.strip()
        assert "|" not in chunk.embedding_text
        assert "#" not in chunk.embedding_text
        assert chunk.chunk_id.startswith(chunk.so_hieu)
        if chunk.part_total > 1:
            assert chunk.sibling_expand


def test_chunk_id_khong_trung(every_chunk):
    identifiers = [chunk.chunk_id for chunk in every_chunk]
    assert len(identifiers) == len(set(identifiers))


def test_breadcrumb_embed_khong_vuot_tran(every_chunk):
    for chunk in every_chunk:
        assert count_tokens(chunk.breadcrumb_embed) <= BREADCRUMB_HARD_CAP or (
            chunk.breadcrumb_level == LEVEL_NONE
        )


def test_chunking_la_ham_thuan():
    """NT-8: cùng đầu vào cho cùng đầu ra byte-for-byte."""
    md_text = (MD_DIR / f"{ND293}.md").read_text(encoding="utf-8")
    first = [json.dumps(chunk.to_row(), ensure_ascii=False) for chunk in chunk_markdown(md_text)]
    second = [json.dumps(chunk.to_row(), ensure_ascii=False) for chunk in chunk_markdown(md_text)]
    assert first == second


# ==========================================================================
# FIXTURE BẮT BUỘC CỦA SPEC
# ==========================================================================

def test_f1_cau_dan_dieu_3_tncn_nhan_ban_va_giu_menh_de_tru(corpus):
    """Câu dẫn phải vào MỌI chunk con và không được đánh rơi vế ``trừ``."""
    chunks = pick(corpus[TNCN], dieu_so="3")
    assert len(chunks) >= 10
    for chunk in chunks:
        assert "trừ thu nhập được miễn thuế" in chunk.content
        assert "trừ thu nhập được miễn thuế" in chunk.embedding_text


def test_f2_doan_sau_diem_cuoi_nam_o_manh_cuoi(corpus):
    """Câu chi phối cả khoản 2 không được tách thành mảnh riêng."""
    chunks = sorted(
        pick(corpus[TNCN], dieu_so="7", khoan_so="2"), key=lambda c: c.part_index
    )
    assert chunks
    governing = "không áp dụng cách tính thuế quy định tại khoản này"
    assert governing in chunks[-1].content
    if len(chunks) > 1:
        assert to_plain(chunks[-1].body_content) != governing


def test_f3_bang_dieu_9_tncn_gon_trong_mot_chunk(corpus):
    tables = [chunk for chunk in corpus[TNCN] if chunk.has_table]
    assert len(tables) == 1
    chunk = tables[0]
    assert (chunk.dieu_so, chunk.khoan_so) == ("9", "2")
    assert chunk.part_total == 1
    assert "|" in chunk.content and "|" not in chunk.embedding_text
    assert chunk.table_id == "112/VBHN-VPQH#tbl1"


def test_f4_summary_bang_nd293_va_rang_buoc_phu_10e(corpus):
    tables = [chunk for chunk in corpus[ND293] if chunk.has_table]
    assert len(tables) == 1
    chunk = tables[0]
    assert chunk.table_id == "293/2025/NĐ-CP#tbl1"
    assert "Vùng I: mức lương tối thiểu tháng là 5.310.000 đồng/tháng" in chunk.table_summary
    assert "đồng/tháng)" not in chunk.table_summary, "đơn vị phải tách khỏi tên cột"

    header, rows = parse_table(chunk.table_raw)
    haystack = _norm(f"{chunk.table_summary} {chunk.table_search_text}").lower()
    for cell in all_cells(header, rows):
        if "Đơn vị" in cell:
            continue
        assert _norm(cell).lower() in haystack, f"ô {cell!r} vô hình với cả hai kênh"
    assert "5310000" in chunk.table_search_text, "thiếu biến thể bỏ dấu phân cách nghìn"


def test_f5_gop_dieu_liet_ke_ngan(corpus):
    """Nhánh 1: Điều ngắn gộp một chunk, vẫn trích dẫn được tới cấp Khoản."""
    merged = [chunk for chunk in corpus[BHXH] if chunk.khoan_range]
    assert merged, "phải có Điều được gộp ở nhánh 1"
    for chunk in merged:
        assert re.fullmatch(r"\d+[a-zđ]?-\d+[a-zđ]?", chunk.khoan_range)
        assert chunk.khoan_so is None
        assert "##### Khoản" in chunk.content
        assert "#" not in chunk.embedding_text
        assert chunk.part_total == 1


def test_f6_dieu_3_bhxh_than_tu_du_van_trong_ngan_sach(corpus):
    """Thân đã tự đủ nghĩa; breadcrumb chỉ lấp phần ngân sách còn thừa."""
    chunks = pick(corpus[BHXH], dieu_so="3")
    assert chunks
    for chunk in chunks:
        assert chunk.token_count <= BUDGET
        assert count_tokens(chunk.breadcrumb_embed) <= BREADCRUMB_HARD_CAP
        assert "Bảo hiểm xã hội" in chunk.content


def test_f7_phu_luc_muc_28_tang_5_lap_tien_to(corpus):
    chunks = sorted(
        [chunk for chunk in corpus[ND293] if "#m28" in chunk.chunk_id],
        key=lambda c: c.part_index,
    )
    assert len(chunks) >= 2
    prefix = "Vùng I, thuộc Thành phố Hồ Chí Minh, gồm"
    for chunk in chunks:
        assert chunk.split_tier == 5
        assert chunk.body_embed.startswith(prefix), chunk.body_embed[:80]


def test_f8_tang_3_cat_theo_dau_cham_phay(every_chunk):
    """Dấu ``;`` là ranh giới liệt kê sạch hơn ranh giới câu."""
    tier3 = [chunk for chunk in every_chunk if chunk.split_tier == 3]
    assert tier3, "corpus phải có ít nhất một mảnh đi qua tầng 3"
    for chunk in tier3:
        # Điều kiện kích hoạt tầng 3 nằm ở ĐƠN VỊ, không ở từng mảnh: mảnh
        # cuối của một chuỗi liệt kê tất nhiên không còn dấu ``;`` nào.
        assert chunk.unit_content.count(";") >= 2


def test_f9_ghep_content_cac_manh_bang_ban_goc(every_chunk):
    """NT-2 áp cho đơn vị dài nhất corpus, không chỉ trung bình."""
    longest = max(every_chunk, key=lambda chunk: len(chunk.unit_content))
    assert len(longest.unit_content) > 3500
    family = sorted(
        [chunk for chunk in every_chunk if chunk.khoan_id == longest.khoan_id],
        key=lambda chunk: chunk.part_index,
    )
    assert len(family) > 1
    joined = _norm(" ".join(chunk.body_content for chunk in family))
    assert joined == _norm(longest.unit_content)


def test_f10_dieu_129_bhxh_breadcrumb_bi_cat_nhung_van_trong_ngan_sach(corpus):
    chunks = pick(corpus[BHXH], dieu_so="129")
    assert chunks
    for chunk in chunks:
        assert count_tokens(chunk.breadcrumb_embed) <= BREADCRUMB_HARD_CAP
        assert chunk.token_count <= BUDGET


def test_f11_dieu_khong_co_khoan(corpus):
    chunks = pick(corpus[TNCN], dieu_so="21")
    assert chunks
    for chunk in chunks:
        assert chunk.khoan_so is None
        assert chunk.branch == 4
        assert chunk.chunk_id.startswith("112/VBHN-VPQH#d21p")


def test_f12_boilerplate_bi_loai_khoi_index(every_chunk):
    target = "chính phủ quy định chi tiết điều này."
    hits = [chunk for chunk in every_chunk if _norm(chunk.body_content).lower() == target]
    assert len(hits) >= 30
    assert not any(chunk.is_indexed for chunk in hits)
    assert all(chunk.content.strip() for chunk in hits), "vẫn phải lưu đầy đủ content"


def test_f13_khoan_trung_nhung_co_noi_dung_that_van_duoc_index(every_chunk):
    hits = [
        chunk
        for chunk in every_chunk
        if "hưởng bảo hiểm xã hội một lần bao gồm" in _norm(chunk.body_content).lower()
    ]
    assert len(hits) >= 2
    assert all(chunk.is_indexed for chunk in hits)


# ==========================================================================
# NT-5 — hồi quy cho phủ định
# ==========================================================================

def test_cau_dan_co_phu_dinh_giu_du_marker_hoac_bi_bo_han(every_chunk):
    """Giữ đầy đủ hoặc bỏ hẳn — không bao giờ giữ bản đã mất vế phủ định.

    Spec kỳ vọng 2 ca rơi bậc C, đo bằng tỉ lệ 3,2 ký tự/token. Tokenizer thật
    cho ~5 ký tự/token nên gần như mọi câu dẫn vừa ``STEM_MAX_TOKENS`` và ở lại
    bậc A — tức mệnh đề phủ định được giữ **nguyên vẹn**, đúng tinh thần NT-5
    và tốt hơn bị bỏ. Test khẳng định đúng bất biến đó, không khẳng định con số
    đã đo trên tỉ lệ sai.
    """
    with_negation = [
        chunk
        for chunk in every_chunk
        if chunk.leadin_full and find_negations(chunk.leadin_full)
    ]
    assert len(with_negation) >= 10, "corpus phải có câu dẫn chứa phủ định"
    for chunk in with_negation:
        if not chunk.leadin_used:
            continue  # bậc C: bỏ hẳn, hợp lệ
        assert set(find_negations(chunk.leadin_full)) <= set(
            find_negations(chunk.leadin_used)
        ), chunk.chunk_id


def test_stem_rut_gon_khong_danh_roi_marker_phu_dinh(every_chunk):
    for chunk in every_chunk:
        if not chunk.stem:
            continue
        assert set(find_negations(chunk.stem_full)) <= set(find_negations(chunk.stem))


# ==========================================================================
# TASK 15 — bẫy metadata của LlamaIndex
# ==========================================================================

def test_llamaindex_khong_chen_metadata_vao_chuoi_embed(every_chunk):
    from llama_index.core.schema import MetadataMode

    from rag.ingestion.chunking import to_text_node

    for chunk in every_chunk[:20]:
        node = to_text_node(chunk.to_row())
        assert node.get_content(MetadataMode.EMBED) == chunk.content
        assert node.text == chunk.content
        assert node.get_content(MetadataMode.LLM) != chunk.content
