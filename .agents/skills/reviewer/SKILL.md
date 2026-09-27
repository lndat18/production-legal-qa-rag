---
name: reviewer
description: Review kiến trúc, logic, security và scalability của PR đã pass checks so với spec; comment PASS hoặc REVISE, không merge.
---

# Reviewer workflow

Đọc `$coding-convention`, spec được chỉ định và PR diff. Tập trung vào kiến trúc, logic, security, scalability, code smell, duplication, naming, typing, maintainability, technical debt; chỉ flag `test-coverage` khi một case quan trọng đã ghi rõ trong spec nhưng chưa có test. Không đánh giá kết quả test/CI.

Chỉ review khi có PR và spec. Xác nhận `gh auth status`, rồi `gh pr checks <PR>`; nếu checks chưa PASS hay không xác minh được, dừng, không comment. Lấy thay đổi chính thức bằng `gh pr diff <PR>`, không dùng working tree để quyết định phạm vi.

Không sửa code/test/CI/config, push, hay merge. Mỗi finding có dạng `file | dòng | loại (architecture/security/scalability/smell/test-coverage/...) | mô tả cụ thể`; với `test-coverage`, nêu case trong spec. Post một PR comment `REVISE` hoặc `PASS` (tóm tắt phạm vi đã đối chiếu). Sau `PASS`, nhắc người dùng merge thủ công, ví dụ `gh pr merge <PR> --squash --delete-branch`.
