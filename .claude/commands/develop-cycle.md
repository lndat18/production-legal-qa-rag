---
description: Chạy vòng lặp developer → tester → reviewer cho một spec cụ thể
argument-hint: <đường dẫn spec.md> <tên branch>
---
Bạn là orchestrator cho quy trình implement code từ spec. Input: $ARGUMENTS (spec path). Giả định spec.md đã được chốt xong cùng agent `architect` trước khi chạy
command này.

## Quy trình

1. **Implement**: Gọi subagent `developer` (Task tool) — đọc spec tại đường dẫn cung cấp
   (kèm feedback tích lũy từ vòng trước nếu có), implement/sửa code, `commit local` vào
   branch chỉ định. Developer KHÔNG bao giờ push hay mở PR.
2. **Test & mở PR — Vòng lặp A**: Gọi subagent `tester` với diff hiện tại + đường dẫn
   spec — viết/cập nhật test, push branch, `gh pr create` (chỉ lần đầu; các vòng sau —
   kể cả commit sửa lỗi của developer — đều do tester push lên cùng PR), rồi theo dõi
   job `checks` trên CI (`gh pr checks --watch`).
   - `checks` fail → tăng biến đếm; nếu < 3 quay lại bước 1 với feedback của tester; nếu
     ≥ 3 → dừng, báo user cần can thiệp thủ công.
   - `checks` pass → sang bước 3.
3. **Review (local) — Vòng lặp B**: Gọi subagent `reviewer` trên PR đã mở — reviewer tự
   xác nhận `checks` đã pass qua `gh pr checks`, lấy diff qua `gh pr diff`, post feedback
   qua `gh pr comment`.
   - `REVISE` → tăng biến đếm (dùng chung bộ đếm với bước 2); nếu < 3 quay lại bước 1 với
     feedback của reviewer (đọc từ PR comment) — vòng sau BẮT BUỘC qua lại bước 2 (tester
     viết thêm test rồi push, Vòng lặp A có thể lặp lại nếu test mới fail) trước khi quay
     lại bước 3; nếu ≥ 3 → dừng, báo user.
   - `PASS` → reviewer tự merge PR (`gh pr merge --squash --delete-branch`) — kết thúc,
     báo user PR đã được merge vào `main`.

Trường hợp mọi thứ đúng ngay từ đầu (developer implement → tester viết test & mở PR →
`checks` pass lần đầu → `reviewer` PASS lần đầu), toàn bộ quy trình chạy thẳng một lượt,
không có vòng lặp nào.

## Lưu ý

- Chỉ `tester` (mở PR, push commit) và `reviewer` (merge) được thao tác git từ xa;
  `developer` chỉ commit local. Reviewer là gate cuối cùng và tự merge PR khi PASS —
  orchestrator không tự merge PR ở bước nào khác.
- Giới hạn 3 vòng lặp implement↔tester/reviewer để tránh loop vô hạn khi spec mơ hồ hoặc
  convention skill xung đột với spec.
- Nếu developer báo lỗi khi chạy check cục bộ ở bước 1, dừng ngay, báo lỗi cho user thay vì
  gọi tester.
