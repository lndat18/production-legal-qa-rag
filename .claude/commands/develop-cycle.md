---
description: Chạy vòng lặp developer → tester → reviewer cho một spec cụ thể
argument-hint: <đường dẫn spec.md> <tên branch>
---
Bạn là orchestrator cho quy trình implement code từ spec. Input: $ARGUMENTS (spec path và branch name). Giả định spec.md đã được chốt xong cùng agent `architect` trước khi chạy
command này.

## Quy trình

1. **Implement**: Gọi subagent `developer` (Task tool) — đọc spec tại đường dẫn cung cấp
   (kèm feedback tích lũy từ vòng trước nếu có), implement/sửa code, commit vào branch
   chỉ định.
2. **Test**: Gọi subagent `tester` với diff hiện tại + đường dẫn spec.
   - `REVISE` → tăng biến đếm; nếu < 3 quay lại bước 1 với feedback của tester; nếu ≥ 3 →
     dừng, báo user cần can thiệp thủ công.
   - `PASS` → sang bước 3.
3. **Review**: Gọi subagent `reviewer` với diff hiện tại + đường dẫn spec (KHÔNG kèm kết
   quả CI).
   - `REVISE` → tăng biến đếm (dùng chung bộ đếm với bước 2); nếu < 3 quay lại bước 1 với
     feedback của reviewer — vòng sau BẮT BUỘC qua lại bước 2 (tester) trước khi quay lại
     bước 3; nếu ≥ 3 → dừng, báo user.
   - `PASS` → sang bước 4.
4. **Mở PR & xác nhận CI** (chỉ khi cả tester và reviewer đã PASS): push branch, `gh pr create` (nếu chưa có), rồi `gh pr checks <branch> --watch` (hoặc poll bằng `gh run list`
   nếu `--watch` không hỗ trợ).
   - CI pass → in "✅ Sẵn sàng merge PR #<số>. Vui lòng review và merge thủ công."
   - CI fail dù local đã PASS (khả năng cao do khác biệt môi trường) → dừng, in log lỗi,
     báo user cần kiểm tra thủ công (KHÔNG tự động quay lại vòng lặp agent).

## Lưu ý

- Không tự động merge PR ở bất kỳ bước nào.
- Giới hạn 3 vòng lặp implement↔tester/reviewer để tránh loop vô hạn khi spec mơ hồ hoặc
  convention skill xung đột với spec.
- Nếu developer báo lỗi khi chạy check cục bộ ở bước 1, dừng ngay, báo lỗi cho user thay vì
  gọi tester.
