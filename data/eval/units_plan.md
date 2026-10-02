# Kế hoạch đơn vị sinh testset

50 đơn vị, 720,573 ký tự, ước lượng ~4.16M token (5,5 token/ký tự + 4K mỗi đơn vị, sai số ±30%).

Sắp xếp từ nhỏ đến lớn (thứ tự ưu tiên chạy); `#` là số thứ tự đơn vị trong văn bản (khoá của file `units/` và của `--only`), không đổi khi sắp xếp.

| Thứ tự chạy | Văn bản | # | Đơn vị | Ký tự | Ước lượng token | Token cộng dồn |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1 | Văn bản hợp nhất bộ luật lao động.md | 13 | Chương XIV. GIẢI QUYẾT TRANH CHẤP LAO ĐỘNG — Mục 2. THẨM QUYỀN VÀ TRÌNH TỰ GIẢI QUYẾT TRANH CHẤP LAO ĐỘNG CÁ NHÂN | 6,000 | 37K | 0.04M |
| 2 | Luật bảo hiểm y tế.md | 6 | Chương IX + Chương X | 6,361 | 38K | 0.08M |
| 3 | Luật bảo hiểm xã hội.md | 9 | Chương VII. QUỸ BẢO HIỂM XÃ HỘI | 6,451 | 39K | 0.12M |
| 4 | Luật thuế thu nhập cá nhân.md | 3 | Chương III + Chương IV | 7,254 | 43K | 0.16M |
| 5 | Luật bảo hiểm xã hội.md | 11 | Chương X. QUẢN LÝ NHÀ NƯỚC VỀ BẢO HIỂM XÃ HỘI | 7,294 | 44K | 0.20M |
| 6 | Luật bảo hiểm xã hội.md | 12 | Chương XI. ĐIỀU KHOẢN THI HÀNH | 7,485 | 45K | 0.25M |
| 7 | Văn bản hợp nhất bộ luật lao động.md | 14 | Chương XIV Mục 3 + Chương XIV Mục 4 | 7,544 | 45K | 0.29M |
| 8 | Văn bản hợp nhất bộ luật lao động.md | 4 | Chương III Mục 4 + Chương III Mục 5 | 7,856 | 47K | 0.34M |
| 9 | Văn bản hợp nhất bộ luật lao động.md | 12 | Chương XIV. GIẢI QUYẾT TRANH CHẤP LAO ĐỘNG — Mục 1. NHỮNG QUY ĐỊNH CHUNG VỀ GIẢI QUYẾT TRANH CHẤP LAO ĐỘNG | 8,596 | 51K | 0.39M |
| 10 | Luật bảo hiểm xã hội.md | 10 | Chương VIII + Chương IX | 8,627 | 51K | 0.44M |
| 11 | Văn bản hợp nhất bộ luật lao động.md | 9 | Chương IX + Chương X | 8,742 | 52K | 0.50M |
| 12 | Văn bản hợp nhất bộ luật lao động.md | 15 | Chương XIV. GIẢI QUYẾT TRANH CHẤP LAO ĐỘNG — Mục 5. ĐÌNH CÔNG | 9,158 | 54K | 0.55M |
| 13 | Văn bản hợp nhất bộ luật lao động.md | 7 | Chương VII. THỜI GIỜ LÀM VIỆC, THỜI GIỜ NGHỈ NGƠI | 9,550 | 56K | 0.61M |
| 14 | Văn bản hợp nhất bộ luật lao động.md | 1 | Chương I. NHỮNG QUY ĐỊNH CHUNG | 10,553 | 62K | 0.67M |
| 15 | Văn bản hợp nhất bộ luật lao động.md | 6 | Chương VI. TIỀN LƯƠNG | 10,828 | 63K | 0.73M |
| 16 | Điều kiện lao động và quan hệ lao động.md | 9 | Chương X. NHỮNG QUY ĐỊNH RIÊNG ĐỐI VỚI LAO ĐỘNG LÀ NGƯỜI GIÚP VIỆC GIA ĐÌNH | 10,917 | 64K | 0.80M |
| 17 | Luật bảo hiểm y tế.md | 4 | Chương V + Chương VI | 10,928 | 64K | 0.86M |
| 18 | Điều kiện lao động và quan hệ lao động.md | 6 | Chương VII. THỜI GIỜ LÀM VIỆC, THỜI GIỜ NGHỈ NGƠI | 10,974 | 64K | 0.93M |
| 19 | Luật bảo hiểm xã hội.md | 4 | Chương V. BẢO HIỂM XÃ HỘI BẮT BUỘC — Mục 1. CHẾ ĐỘ ỐM ĐAU | 11,006 | 64K | 0.99M |
| 20 | Luật bảo hiểm y tế.md | 5 | Chương VII + Chương VIII | 11,209 | 65K | 1.06M |
| 21 | Điều kiện lao động và quan hệ lao động.md | 7 | Chương VIII. KỶ LUẬT LAO ĐỘNG, TRÁCH NHIỆM VẬT CHẤT | 11,327 | 66K | 1.12M |
| 22 | Luật thuế thu nhập cá nhân.md | 1 | Chương I. NHỮNG QUY ĐỊNH CHUNG | 11,419 | 66K | 1.19M |
| 23 | Văn bản hợp nhất bộ luật lao động.md | 8 | Chương VIII. KỶ LUẬT LAO ĐỘNG, TRÁCH NHIỆM VẬT CHẤT | 11,835 | 69K | 1.26M |
| 24 | Điều kiện lao động và quan hệ lao động.md | 2 | Chương IV Mục 1 + Chương IV Mục 2 | 12,044 | 70K | 1.33M |
| 25 | Luật thuế thu nhập cá nhân.md | 2 | Chương II. CĂN CỨ TÍNH THUẾ ĐỐI VỚI CÁ NHÂN CƯ TRÚ | 12,230 | 71K | 1.40M |
| 26 | Luật bảo hiểm y tế.md | 1 | Chương I. NHỮNG QUY ĐỊNH CHUNG | 12,378 | 72K | 1.47M |
| 27 | Luật bảo hiểm xã hội.md | 7 | Chương V. BẢO HIỂM XÃ HỘI BẮT BUỘC — Mục 4. CHẾ ĐỘ TỬ TUẤT | 12,733 | 74K | 1.55M |
| 28 | Văn bản hợp nhất bộ luật lao động.md | 11 | Chương XII + Chương XIII | 13,035 | 75K | 1.62M |
| 29 | Văn bản hợp nhất bộ luật lao động.md | 2 | Chương II + Chương III Mục 1 | 13,195 | 76K | 1.70M |
| 30 | Văn bản hợp nhất bộ luật lao động.md | 16 | Chương XV + Chương XVI + Chương XVII | 13,328 | 77K | 1.77M |
| 31 | Điều kiện lao động và quan hệ lao động.md | 10 | Chương XI. GIẢI QUYẾT TRANH CHẤP LAO ĐỘNG — Mục 1. HÒA GIẢI VIÊN LAO ĐỘNG | 13,790 | 79K | 1.85M |
| 32 | Luật bảo hiểm y tế.md | 3 | Chương III + Chương IV | 14,924 | 86K | 1.94M |
| 33 | Luật bảo hiểm y tế.md | 2 | Chương II. ĐỐI TƯỢNG, MỨC ĐÓNG, TRÁCH NHIỆM VÀ PHƯƠNG THỨC ĐÓNG BẢO HIỂM Y TẾ | 16,992 | 97K | 2.04M |
| 34 | Điều kiện lao động và quan hệ lao động.md | 8 | Chương IX. LAO ĐỘNG NỮ VÀ BẢO ĐẢM BÌNH ĐẲNG GIỚI | 17,093 | 98K | 2.14M |
| 35 | Điều kiện lao động và quan hệ lao động.md | 11 | Chương XI. GIẢI QUYẾT TRANH CHẤP LAO ĐỘNG — Mục 2. HỘI ĐỒNG TRỌNG TÀI LAO ĐỘNG | 17,814 | 101K | 2.24M |
| 36 | Văn bản hợp nhất bộ luật lao động.md | 10 | Chương XI. NHỮNG QUY ĐỊNH RIÊNG ĐỐI VỚI LAO ĐỘNG CHƯA THÀNH NIÊN VÀ MỘT SỐ LAO ĐỘNG KHÁC | 17,837 | 102K | 2.34M |
| 37 | Luật bảo hiểm xã hội.md | 1 | Chương I. NHỮNG QUY ĐỊNH CHUNG | 17,978 | 102K | 2.44M |
| 38 | Điều kiện lao động và quan hệ lao động.md | 3 | Chương IV. CHO THUÊ LẠI LAO ĐỘNG — Mục 3. ĐIỀU KIỆN, THẨM QUYỀN, TRÌNH TỰ, THỦ TỤC CẤP, GIA HẠN, CẤP LẠI, THU HỒI GIẤY PHÉP VÀ DANH MỤC CÔNG VIỆC ĐƯỢC THỰC HIỆN CHO THUÊ LẠI LAO ĐỘNG | 18,118 | 103K | 2.55M |
| 39 | Điều kiện lao động và quan hệ lao động.md | 5 | Chương VI. TIỀN LƯƠNG | 18,212 | 104K | 2.65M |
| 40 | Điều kiện lao động và quan hệ lao động.md | 12 | Chương XI Mục 3 + Chương XI Mục 4 + Chương XI | 18,235 | 104K | 2.76M |
| 41 | Luật bảo hiểm xã hội.md | 2 | Chương II. QUYỀN, TRÁCH NHIỆM CỦA CƠ QUAN, TỔ CHỨC, CÁ NHÂN VỀ BẢO HIỂM XÃ HỘI VÀ TỔ CHỨC THỰC HIỆN BẢO HIỂM XÃ HỘI | 19,425 | 110K | 2.87M |
| 42 | Luật bảo hiểm xã hội.md | 8 | Chương VI. BẢO HIỂM XÃ HỘI TỰ NGUYỆN | 21,325 | 121K | 2.99M |
| 43 | Văn bản hợp nhất bộ luật lao động.md | 3 | Chương III Mục 2 + Chương III Mục 3 | 22,589 | 128K | 3.12M |
| 44 | Điều kiện lao động và quan hệ lao động.md | 4 | Chương IV Mục 4 + Chương V | 23,170 | 131K | 3.25M |
| 45 | Luật bảo hiểm xã hội.md | 5 | Chương V. BẢO HIỂM XÃ HỘI BẮT BUỘC — Mục 2. CHẾ ĐỘ THAI SẢN | 24,615 | 139K | 3.39M |
| 46 | Điều kiện lao động và quan hệ lao động.md | 1 | Chương I + Chương II + Chương III | 25,466 | 144K | 3.53M |
| 47 | Luật bảo hiểm xã hội.md | 6 | Chương V. BẢO HIỂM XÃ HỘI BẮT BUỘC — Mục 3. CHẾ ĐỘ HƯU TRÍ | 27,064 | 152K | 3.68M |
| 48 | Luật bảo hiểm xã hội.md | 3 | Chương III + Chương IV | 27,337 | 154K | 3.84M |
| 49 | Văn bản hợp nhất bộ luật lao động.md | 5 | Chương IV + Chương V | 28,657 | 161K | 4.00M |
| 50 | Quy định mức lương tối thiểu.md | 1 | Toàn văn | 29,075 | 163K | 4.16M |
