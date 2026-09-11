# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.x; PyTorch/Ultralytics runtime; Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `class_id=468`, `class_name="cab"`, `rank=1`, `score=0.510915`, `taxonomy_name="ImageNet-1K"`.
- Record này mô tả toàn ảnh như thế nào?
  - Đây là prediction cấp ảnh, không phải nhãn cho từng vật thể. Record cho biết hình ảnh `traffic` được model xếp hạng đầu tiên là lớp `cab` với độ tin cậy 0.510915. Nói cách khác, mô hình suy đoán toàn bộ ảnh gần với cảnh “cab” hơn các lớp khác, nhưng đó chỉ là điểm số của model chứ không phải ground truth do người gán nhãn xác nhận.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  - Class list đến từ taxonomy đi kèm với checkpoint: đối với `yolo11n-cls.pt` là `ImageNet-1K`. Tức là danh sách các lớp là tập hợp được định nghĩa sẵn bởi mô hình và dữ liệu huấn luyện, không phải model “tự nghĩ ra” tên lớp trong quá trình chạy.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - `class_id` là mã số trong taxonomy; `class_name` là tên đọc hiểu của người dùng; `taxonomy_name` cho biết lớp thuộc hệ thống phân loại nào. Cần cả ba để tránh nhầm lẫn giữa các taxonomy khác nhau. Ví dụ, cùng một tên lớp có thể khác ID ở các bộ taxonomy; `cab` trong ImageNet-1K khác hoàn toàn với định nghĩa lớp trong COCO hoặc taxonomy khác.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  - Guideline cần quy định một nguyên tắc “single-label vs multi-label” rõ ràng. Với phân loại ảnh, nếu một bức ảnh có nhiều chủ thể, phải xác định nhãn chính dựa trên đối tượng chiếm ưu thế hoặc cảnh chính. Nếu không chắc, cần có quy tắc ưu tiên hoặc escalate cho reviewer để tránh gán nhãn theo cảm tính.
- Vì sao model score không phải ground truth?
  - Score là xác suất mà model gán cho từng lớp dựa trên dữ liệu huấn luyện, không phải kết quả do con người kiểm tra và đồng thuận. Ground truth thuộc về quy trình guideline → gán nhãn → QC/rework, còn score chỉ là một “signal” để hỗ trợ xem xét, không thể thay thế cho nhãn chuẩn.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  - Ví dụ: `class_name="person"`, `score=0.912625`, `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`, `bbox_width=113.58`, `bbox_height=279.68`.
- Diễn giải vị trí box bằng lời:
  - Hộp này bắt đầu ở tọa độ x≈385.33, y≈69.24 và kết thúc ở x≈498.92, y≈348.92 trên ảnh có kích thước 640x427 pixel. Với kích thước khoiảng 113.58 pixel theo chiều ngang và 279.68 pixel theo chiều dọc, box bao phủ một người ở góc phải của ảnh gần như toàn thân.
- So sánh số prediction ở hai threshold:
  - Dữ liệu cho thấy ở ngưỡng 0.35 có 53 prediction trên tất cả các sample; ở ngưỡng 0.5 còn 33 prediction; ở 0.7 chỉ còn 15 prediction. Theo từng sample: `traffic` 26 → 18 → 9, `kitchen` 11 → 6 → 3, `dining` 16 → 9 → 3.
  - Như vậy, tăng threshold làm giảm số box được giữ lại và giảm “nhiễu” nhưng cũng có nguy cơ bỏ sót object yếu.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Khi threshold thấp, reviewer phải xem nhiều box hơn, bao gồm nhiều prediction có độ tin cậy thấp hoặc có thể là false positive. Khi threshold cao, độ bao phủ giảm và khối lượng review nhẹ hơn, nhưng đánh giá có thể thiếu các vật thể nhỏ/đậm độ mơ hồ.
- Đề xuất một quy tắc box chặt:
  - Box nên là bounding box “chặt” nhất quanh phần vật thể đang thấy, không bao phủ nhiễu nền, không quá rộng, không quá dài. Nói cách khác, box phải bao gồm toàn bộ phần visible object và tối thiểu vùng nền. Nếu vật thể bị lấn ra ngoài khung hình, box phải dừng ở mép ảnh chứ không suy đoán phần ẩn.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - Cần có guideline riêng cho “truncated/occluded object”: định nghĩa box chỉ bao quanh vùng visible, không dựng phần bị ẩn. Nếu phần che khuất quá lớn, hoặc không chắc có phải là một instance riêng hay bị chồng lên bởi object khác, cần escalate cho reviewer để quyết định theo quy tắc thống nhất.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - Ví dụ: `instance_id="kitchen-001"`, `class_name="bus"`, `score=0.925745`, `polygon_point_count=120` và một phần `polygon_xy` chứa nhiều cặp tọa độ theo contour của đối tượng.
- Polygon bổ sung chi tiết gì so với box?
  - Box chỉ là hình chữ nhật bao quanh đối tượng, còn polygon mô tả chính xác đường viền của vật thể, kể cả các góc, lồi lõm và hình dạng không đều. Điều này đặc biệt quan trọng khi đối tượng có form phức tạp hoặc khi có nhiều phần che khuất/trùng lặp với nền.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` dùng để phân biệt từng đối tượng riêng lẻ trong cùng một ảnh, giúp nhóm polygon, box, class và score cho đúng instance. Nó không phải `class_id` (mã lớp), không phải `track_id` (theo dõi qua frame), cũng không phải `sample_id` (mã ảnh/đoạn ảnh), mà là ID của instance trong ảnh đó.
- Đề xuất một quy tắc biên mask:
  - Biên mask nên theo đúng silhouette có thể nhìn thấy của vật thể, nối liền thành mặt kín; không được “đánh thẳng” thành hình chữ nhật hay đi quá ra ngoài vùng thực tế. Với các vùng tiếp xúc hoặc chồng lấn, nên giữ mask của từng instance riêng biệt và tránh “bám” vào object lân cận.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Cần định nghĩa rõ: vùng mờ nên chỉ giữ phần rõ ràng, không suy diễn phần bị che; vùng tiếp xúc giữa các instance cần phân tách theo contour rõ nhất hoặc đánh dấu “ambiguous”; trường hợp không chắc chắn, phải escalate cho reviewer/annotator để quyết định bằng quy tắc chung thay vì tự suy đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh, định dạng single-label theo taxonomy (ví dụ: `class_id`, `class_name`, `taxonomy_name`) | Một bức ảnh có nhiều đối tượng, nhầm taxonomy, scene không rõ ràng, nhãn chủ đạo không thống nhất | Chọn nhãn phù hợp với guideline và đối tượng chính của ảnh; nếu không chắc thì dùng quy tắc ưu tiên hoặc báo reviewer | Kiểm tra class ID, taxonomy, label đúng với image-level guideline và không có sai lệch taxonomy |
| Phát hiện vật thể | Class + box cho từng object, định dạng `bbox_xyxy` / `bbox_width` / `bbox_height` theo pixel | Nhiều box chồng chéo, box quá rộng, object qúa nhỏ, bị cắt mép, che khuất | Vẽ box chặt quanh phần visible object; không suy đoán phần ẩn; lưu class đúng theo COCO | Xem box tightness, class correctness, threshold phù hợp và đánh dấu trường hợp cần escalation |
| Instance segmentation | Class + polygon cho từng instance, có `instance_id`, số điểm polygon, contour rõ ràng | Polygon quá rộng, mask lẫn vào vật thể khác, vùng mờ/tiếp xúc, phần che khuất không rõ | Vẽ contour theo visible silhouette, giữ instance riêng biệt, đánh dấu vùng không chắc | Kiểm tra biên mask, instance separation, tính nhất quán còn xem có cần rework hay escalation |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  - Chỉ sử dụng ảnh công khai đã được phê duyệt, không tải dữ liệu cá nhân, khách hàng, nội bộ hoặc nhạy cảm vào repository/report; không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu nhận dạng vào `REPORT.md`, JSON hay output.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  - Lab Coach / GV giảng dạy hoặc người phụ trách project, kèm mô tả rõ ảnh/data đang gặp vấn đề và yêu cầu hướng dẫn tiếp theo trước khi tiếp tục xử lý.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
