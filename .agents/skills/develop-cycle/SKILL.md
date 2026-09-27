---
name: develop-cycle
description: Điều phối tuần tự developer → tester → reviewer cho một spec đã chốt; dùng khi người dùng yêu cầu toàn bộ lifecycle gồm local gates, CI và PR.
---

# Develop cycle

Chỉ dùng khi user cung cấp đúng `<spec-path> <branch>` và yêu cầu lifecycle đầy đủ. Nếu thiếu/không tách an toàn được, yêu cầu lại theo đúng dạng đó.

## Preflight

Xác nhận spec tồn tại; branch tồn tại hoặc tạo an toàn từ `main`; worktree không có thay đổi ngoài task; các roles `developer`, `tester`, `reviewer` sẵn sàng; và `gh auth status` trước các bước remote. Không đổi branch, stash hoặc loại bỏ thay đổi của user khi worktree bẩn.

Lập ledger trong hội thoại: spec path, branch, PR (ban đầu rỗng), SHA mới nhất, feedback tích luỹ, `ci_feedback_count = 0`, `design_feedback_count = 0`. Gọi roles tuần tự, không song song vì cùng thay đổi một branch.

## Cycle

1. Gọi `developer` với spec, branch và feedback tích luỹ. Nếu hard local gate fail hoặc blocker, dừng. Lưu SHA và migration manifest nếu có.
2. Gọi `tester` với spec, branch, SHA, PR hiện tại, migration manifest và feedback cần có test. `CHECKS_FAIL` tăng `ci_feedback_count`; dưới 3 thì quay lại developer, đạt 3 thì dừng. `CHECKS_PASS` lưu PR/SHA và chuyển reviewer. `BLOCKED` thì dừng.
3. Gọi `reviewer` với spec, branch, PR, SHA. `REVISE` tăng `design_feedback_count`; dưới 3 thì đưa feedback vào ledger và quay lại developer, đạt 3 thì dừng. `PASS` kết thúc; reviewer đã comment nhưng không merge. `BLOCKED` thì dừng.

Chỉ tester được push/mở PR; developer chỉ commit local; reviewer không merge; orchestrator không tự push/tạo PR/merge. Mỗi handoff phải nêu trạng thái, SHA, PR, feedback định dạng chuẩn và hành động kế tiếp. Kết quả cuối nêu spec, branch, PR, SHA cuối, hai counters, trạng thái checks, verdict reviewer; khi PASS, nhắc PR chờ user merge thủ công.
