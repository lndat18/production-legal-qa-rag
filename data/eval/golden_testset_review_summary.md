Kết quả chọn: đủ 180 mẫu (162 single-hop + 18 multi-hop); forced_fill: 23 mẫu.

## Kết quả rà soát

- Keep: 157; drop: 46.
- Đếm theo reason_code:
  - `answer_unsupported`: 13
  - `shallow_topic`: 12
  - `bad_multihop`: 5
  - `mechanical`: 5
  - `duplicate`: 4
  - `language`: 3
  - `not_self_contained`: 3
  - `transitional_clause`: 1

## Theo source_document

| Source document | Keep, selected | Keep, not selected | Drop, selected | Drop, not selected |
|---|---:|---:|---:|---:|
| Luật bảo hiểm xã hội.md | 46 | 0 | 6 | 6 |
| Luật bảo hiểm y tế.md | 18 | 0 | 1 | 2 |
| Luật thuế thu nhập cá nhân.md | 6 | 0 | 2 | 0 |
| Quy định mức lương tối thiểu.md | 8 | 0 | 0 | 1 |
| Văn bản hợp nhất bộ luật lao động.md | 37 | 0 | 9 | 3 |
| Điều kiện lao động và quan hệ lao động.md | 42 | 0 | 5 | 11 |

## Theo synthesizer_name

| Synthesizer | Keep, selected | Keep, not selected | Drop, selected | Drop, not selected |
|---|---:|---:|---:|---:|
| `multi_hop_specific_query_synthesizer` | 15 | 0 | 3 | 5 |
| `single_hop_specific_query_synthesizer` | 142 | 0 | 20 | 18 |

## Chất lượng của mẫu keep

| Điểm | Số mẫu |
|---:|---:|
| 1 | 0 |
| 2 | 12 |
| 3 | 49 |
| 4 | 71 |
| 5 | 25 |

## Nhóm trùng

- Giữ 53 (`234524264913`); loại trùng: 47 (`0ec82903709e`).
- Giữ 125 (`411e5e85a625`); loại trùng: 128 (`e7695b2d82c5`).
- Giữ 23 (`66e4f1ae5602`); loại trùng: 22 (`cc646bccd5d5`).
- Giữ 82 (`f0d6b3269efb`); loại trùng: 84 (`a679960c580c`).

## Ví dụ theo reason_code

- `answer_unsupported` (13): 25 (`a530e572be6a`); 45 (`bbd41e7c3f01`); 49 (`b40fefaeb798`).
- `shallow_topic` (12): 8 (`fdc8a7280b43`); 17 (`81f9928f1e58`); 32 (`2d6cdcc358c4`).
- `bad_multihop` (5): 81 (`d7ee72f0ef3b`); 86 (`78b9b9ca8719`); 106 (`52686d995ca7`).
- `mechanical` (5): 6 (`276080ee9d0f`); 27 (`ef36b11d98eb`); 28 (`77dad74fb3e1`).
- `duplicate` (4): 22 (`cc646bccd5d5`); 47 (`0ec82903709e`); 84 (`a679960c580c`).
- `language` (3): 38 (`acd866ecc412`); 99 (`2f3a1b67bd19`); 156 (`3855d68c4e75`).
- `not_self_contained` (3): 5 (`7fc1c2085c7e`); 55 (`e7a5af075c6c`); 144 (`c37974fa1e9c`).
- `transitional_clause` (1): 11 (`ba758327d168`).

## Ngoại lệ và điểm chưa chắc

- Một số đáp án nêu đúng nguyên tắc nhưng thiếu điều kiện chi tiết; các trường hợp này được giữ với điểm thấp theo quy tắc ưu tiên giữ khi chưa chắc.
- Mẫu multi-hop có ít đoạn `reference_contexts` không bị loại chỉ vì thiếu đoạn. Khi nội dung dẫn Điều khác, đánh giá dựa trên ngữ cảnh có sẵn và phần văn bản markdown đã đối chiếu cho các trường hợp đáng ngờ.
- Với danh sách địa bàn Vùng III ở index 199, đáp án mới nêu một phần danh sách; vẫn giữ điểm thấp thay vì tự bổ sung nội dung.
- Các lý do chính trong quyết định loại trùng đã được gán `duplicate`; nếu đồng thời có lỗi nội dung, ghi ngắn trong `note` khi thích hợp.

## Kiểm tra tệp

- Review: 203 mẫu; index liên tục 0–202; case_id duy nhất: có.
- Candidate: 180 mẫu; single-hop 162, multi-hop 18; case_id duy nhất: True.
- Mỗi candidate khớp nguyên vẹn cả 6 trường với mẫu gốc: có.
- Tập case_id được đánh selected trong review trùng chính xác với tập case_id trong candidate: có.
- Forced_fill: 23 mẫu; phân loại: single_hop_specific_query_synthesizer=20, multi_hop_specific_query_synthesizer=3.
