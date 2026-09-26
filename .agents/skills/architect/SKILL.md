---
name: architect
description: Brainstorm và chốt một *_spec.md cùng người dùng trước khi implement; dùng cho yêu cầu plan, viết hoặc sửa spec.
---

# Architect workflow

Thiết kế ở mức production nhưng không over-engineering: ưu tiên 20% phần lõi quan trọng nhất và giữ phần còn lại đơn giản.

Trước khi đề xuất cấu trúc code/module, đọc `$coding-convention`. Đọc spec hiện có (nếu có) và các spec liên quan để giữ nhất quán.

Chốt cùng người dùng: mục tiêu, phạm vi, tiêu chí hoàn thành và phần không làm; input/output và số liệu thực tế nếu có; tool/công nghệ cụ thể; workflow và quản lý trạng thái. Hỏi các điểm mơ hồ có ảnh hưởng đáng kể trước khi đề xuất; không tự đoán thông tin quan trọng.

Chỉ tạo/chỉnh sửa `*_spec.md`, sau khi các quyết định chính đã được chốt. Với spec mới, đặt file cạnh package nó mô tả. Báo rõ quyết định đã chốt, điểm còn mở và giả định; không implement hoặc sửa source/test/configuration.
