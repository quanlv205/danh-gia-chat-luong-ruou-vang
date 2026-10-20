# Đánh giá chất lượng rượu vang dựa trên thành phần hóa lý

Bài toán phân loại mức chất lượng rượu vang (thấp / trung bình / cao) từ 11 chỉ số hóa lý,
dùng bộ dữ liệu Wine Quality (UCI). So sánh Decision Tree, Bagging, Random Forest, Extra Trees
với hai mô hình cơ sở (lớp đa số, Logistic Regression); khảo sát lỗi OOB và độ quan trọng đặc trưng.

## Cài đặt và chạy

```bash
pip install -r requirements.txt
python wine_quality_rf_v2.py            # gộp vang đỏ + trắng (cấu hình dùng trong báo cáo)
FAST=1 python wine_quality_rf_v2.py     # chạy nhanh để kiểm tra lỗi, KHÔNG dùng số liệu này
```

- Dữ liệu: nếu có sẵn `winequality-red.csv`, `winequality-white.csv` cùng thư mục thì đọc file đó,
  nếu không script tự tải từ UCI (cần Internet).
  Nguồn: https://archive.ics.uci.edu/dataset/186/wine+quality
- Seed cố định `SEED = 42` (chia tập, mô hình, cross-validation) nên chạy lại cho kết quả giống nhau
  trên cùng phiên bản thư viện. Thời gian chạy đầy đủ: vài phút.
- Tùy chọn `red` / `white` để phân tích riêng từng loại (kết quả lưu vào `ket_qua_red/`, `ket_qua_white/`).

## Quy trình trong code

1. Đọc dữ liệu, tạo từ điển dữ liệu và thống kê mô tả.
2. Định lượng vấn đề dữ liệu (thiếu, trùng, ngoại lai, mất cân bằng, rò rỉ) và xử lý, có so sánh trước/sau.
3. EDA: phân phối nhãn, phân phối đặc trưng, tương quan, boxplot theo nhãn.
4. Chia train/test có phân tầng (80/20) trước mọi phép biến đổi.
5. Pipeline cho từng mô hình; cross-validation 5-fold trên train (trung bình ± độ lệch chuẩn, có điểm train
   để nhận diện quá khớp); tập test chỉ đánh giá cuối.
6. Ablation: Decision Tree -> Bagging -> Random Forest -> + class_weight -> + đặc trưng mới -> + tinh chỉnh.
7. Khảo sát `n_estimators`, `max_features`, `max_depth` (lỗi train, OOB, test tham khảo).
8. Tối ưu siêu tham số bằng RandomizedSearchCV trên cross-validation (không dùng tập test).
9. Mô hình cuối chọn theo CV F1 macro; kiểm tra độ ổn định qua nhiều seed, đường cong học tập,
   ma trận nhầm lẫn, phân tích lỗi, độ quan trọng đặc trưng (impurity và permutation).

## Kết quả đầu ra (thư mục `ket_qua/`)

| File | Nội dung |
|---|---|
| `tu_dien_du_lieu.csv`, `thong_ke_mo_ta_truoc_xu_ly.csv` | Từ điển dữ liệu, thống kê mô tả |
| `van_de_du_lieu.csv`, `ngoai_lai_theo_dac_trung.csv`, `so_sanh_truoc_sau_xu_ly.csv` | Vấn đề dữ liệu và xử lý |
| `bang_tong_hop_va_ablation.csv` | CV (TB ± SD), điểm train, test của mọi mô hình và các bước ablation |
| `ket_qua_tim_kiem_sieu_tham_so.csv` | Kết quả RandomizedSearchCV |
| `06_…` đến `09_…` (png, csv) | Khảo sát siêu tham số |
| `phan_tich_loi.csv`, `do_quan_trong_dac_trung.csv` | Phân tích lỗi, độ quan trọng đặc trưng |
| `01_…` đến `12_…` (png) | Các hình dùng trong báo cáo |
| `mo_hinh_cuoi.joblib` | Mô hình cuối đã huấn luyện |

## Phân công và nhật ký đóng góp

| Thành viên | Phần phụ trách | Minh chứng (commit / nhật ký) |
|---|---|---|
| … | … | … |
| … | … | … |

## Phạm vi sử dụng công cụ AI

(Ghi rõ đã dùng công cụ AI tạo sinh cho những phần nào: ví dụ hỗ trợ viết khung mã nguồn và soạn thảo báo cáo;
phần nào do nhóm tự thực hiện/chỉnh sửa. Mỗi thành viên phải giải thích được toàn bộ mã nguồn và nội dung đã nộp.)
