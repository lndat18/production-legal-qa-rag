"""Kiểm thử CLI chạy thử luồng hội thoại nhiều lượt."""

from tools.conversation import _status_message


def test_status_message_covers_all_generation_stages() -> None:
    """CLI phải hiển thị mọi stage mà ``GenerationEvent`` hiện phát."""
    assert _status_message("guardrail") == "Đang kiểm tra an toàn / viết lại câu hỏi..."
    assert _status_message("retrieval") == "Đang tìm văn bản liên quan..."
    assert _status_message("drafting") == "Đang tạo bản nháp câu trả lời..."
    assert _status_message("verification") == "Đang kiểm chứng căn cứ pháp lý..."
    assert _status_message("repairing") == "Đang điều chỉnh câu trả lời..."


def test_status_message_falls_back_for_future_stage() -> None:
    """CLI không được dừng stream chỉ vì source bổ sung stage mới."""
    assert _status_message("future_stage") == "Đang xử lý câu hỏi..."
