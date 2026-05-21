# Thiết kế mã nguồn cho mô hình End-to-End OVOD + OVRD

## 1. Giới thiệu

Đây là tài liệu thiết kế mã nguồn cho mô hình Open-Vocabulary Object Detection (OVOD) và Open-Vocabulary Relation Detection (OVRD) dựa trên kiến trúc Transformer, được mô tả chi tiết trong tài liệu đầu vào. Mục tiêu là xây dựng một hệ thống có khả năng phát hiện đối tượng và quan hệ giữa chúng trong ảnh, ngay cả với các lớp chưa từng thấy trong quá trình huấn luyện.

## 2. Kiến trúc tổng quan

Mô hình sẽ tuân theo kiến trúc End-to-End như sau:

1.  **Backbone thị giác:** Trích xuất đặc trưng hình ảnh đa tỷ lệ.
2.  **Transformer Encoder:** Tạo biểu diễn ngữ cảnh toàn cục từ feature maps.
3.  **Transformer Decoder:** Sử dụng Object Queries và Relation Queries để dự đoán đối tượng và quan hệ. Bao gồm các module SSGA và SGOR.
4.  **Prediction Heads:** Chuyển đổi output của decoder thành các dự đoán cuối cùng về bounding box, class và predicate.
5.  **Hàm Loss:** Tổng hợp các hàm mất mát cho từng thành phần của mô hình.
6.  **Quy trình huấn luyện 2 giai đoạn:** Pre-training (weak supervision) và Fine-tuning (full supervision).

## 3. Cấu trúc thư mục và module

Mã nguồn sẽ được tổ chức theo cấu trúc thư mục sau:

```
. # Thư mục gốc của dự án
├── README.md
├── requirements.txt
├── main.py # Script chính để chạy huấn luyện/đánh giá
├── config.py # File cấu hình cho mô hình và huấn luyện
├── utils/ # Các hàm tiện ích chung
│   ├── __init__.py
│   └── distributed_utils.py # Hỗ trợ huấn luyện phân tán
│   └── box_ops.py # Các phép toán trên bounding box
│   └── misc.py # Các tiện ích khác
├── data/ # Xử lý dữ liệu và dataset
│   ├── __init__.py
│   ├── datasets.py # Định nghĩa các dataset (COCO, LVIS, VG150, GQA, CC3M, CC12M, Flickr30k)
│   ├── data_loaders.py # DataLoader và Sampler
│   └── transforms.py # Các phép biến đổi dữ liệu (augmentation)
├── models/ # Định nghĩa kiến trúc mô hình
│   ├── __init__.py
│   ├── ovod_model.py # Lớp chính của mô hình OVOD+OVRD
│   ├── backbone.py # Visual Backbone (Swin Transformer, ResNet)
│   ├── transformer.py # Transformer Encoder và Decoder
│   ├── prediction_heads.py # Object Head và Relation Head
│   ├── ssga.py # Module Sparse Scene-Graph-Guided Attention
│   ├── sgor.py # Module Scene-Graph-Based Offset Regression
│   └── criterion.py # Định nghĩa các hàm Loss và HungarianMatcher
├── engine.py # Logic huấn luyện và đánh giá
```

## 4. Chi tiết các module chính

### 4.1. `models/ovod_model.py`

Lớp `OVODModel` sẽ là điểm vào chính của mô hình, tích hợp các thành phần con:

-   Khởi tạo `Backbone`, `Transformer`, `PredictionHeads`.
-   Định nghĩa phương thức `forward` để xử lý đầu vào và trả về output của mô hình.

### 4.2. `models/backbone.py`

-   Chứa các lớp cho Visual Backbone như `SwinTransformer` hoặc `ResNet` (có thể sử dụng các implementation có sẵn từ `timm` hoặc `torchvision`).
-   Đảm bảo output là feature maps đa tỷ lệ ({C3, C4, C5}).

### 4.3. `models/transformer.py`

-   Chứa lớp `TransformerEncoder` và `TransformerDecoder` (có thể dựa trên `torch.nn.Transformer` hoặc các implementation từ DETR/Deformable-DETR).
-   `TransformerDecoder` sẽ xử lý cả Object Queries và Relation Queries.

### 4.4. `models/prediction_heads.py`

-   `ObjectHead`: MLP để dự đoán tọa độ box và Linear layer để dự đoán class embedding (trong không gian CLIP text space).
-   `RelationHead`: MLP để dự đoán predicate embedding và cơ chế pointer/attention để gán subject/object từ các object queries đã dự đoán.

### 4.5. `models/ssga.py`

-   Triển khai module Sparse Scene-Graph-Guided Attention.
-   Đây là một trong những phần khó nhất, cần xây dựng scene graph động và custom attention mask.

### 4.6. `models/sgor.py`

-   Triển khai module Scene-Graph-Based Offset Regression.
-   Sử dụng các MLP layer để tinh chỉnh bounding box qua nhiều lớp decoder lặp lại.

### 4.7. `models/criterion.py`

-   Chứa lớp `SetCriterion` để tính toán tổng hợp các hàm loss:
    -   `L_obj_cls`: Focal Loss.
    -   `L_obj_bbox`: L1 Loss + GIoU Loss.
    -   `L_rel_cls`: Focal/BCE trên predicate.
    -   `L_assign`: Hungarian matching cho triplet (predicate, subject box, object box).
    -   `L_vl`: InfoNCE / CLIP loss.
    -   `L_kd`: KL divergence (tùy chọn).
-   `HungarianMatcher`: Thực hiện thuật toán Hungarian matching để gán các dự đoán với ground truth.

### 4.8. `data/datasets.py`

-   Định nghĩa các lớp `Dataset` cho từng loại dữ liệu (COCO, LVIS, Visual Genome, GQA, CC3M, CC12M, Flickr30k).
-   Xử lý việc tải ảnh, annotations và scene graph (nếu có).
-   Đối với pre-training, cần tích hợp scene graph parser (spaCy, FACTUAL) để trích xuất entity và relation từ caption.

## 5. Kế hoạch triển khai (Roadmap)

Tuân thủ lộ trình đề xuất để giảm thiểu độ phức tạp ban đầu và tăng dần tính năng:

-   **Tuần 1–2:** Clone Deformable-DETR làm nền tảng. Thêm Relation Query và một simple relation head. Tập trung vào việc làm cho mô hình cơ bản chạy được với Object Detection và Relation Detection đơn giản.
-   **Tuần 3–4:** Implement Hungarian matching cho triplet (subject, predicate, object). Đây là một bước quan trọng để xử lý việc gán quan hệ chính xác.
-   **Tuần 5–6:** Tích hợp CLIP text encoder và thêm InfoNCE loss (`L_vl`). Bắt đầu thử nghiệm khả năng Open-Vocabulary.
-   **Tuần 7–8:** Implement SSGA với custom sparse attention. Đây là một trong những phần khó nhất, cần cẩn thận trong việc thiết kế và debug.
-   **Tuần 9+:** Fine-tune hyperparameters, chạy thực nghiệm trên VG150 và các dataset khác. Giải quyết các vấn đề về Gradient instability và tối ưu hóa hiệu suất.

## 6. Rủi ro và thách thức

-   **Gradient instability:** Cần điều chỉnh cẩn thận các trọng số `λ` của các hàm loss khác nhau để đảm bảo mô hình hội tụ ổn định.
-   **SSGA và Relation Query Assignment:** Đây là hai module phức tạp nhất, đòi hỏi kỹ thuật cao và có thể không có thư viện sẵn.
-   **Scale dữ liệu pre-training:** Yêu cầu pipeline data loading hiệu quả và khả năng xử lý scene graph parsing tự động.
-   **Open-Vocabulary Generalization:** Cần cân bằng giữa việc fine-tune text encoder và giữ khả năng generalize cho các lớp mới.

## 7. Môi trường phát triển

-   **Ngôn ngữ:** Python 3.8+
-   **Thư viện chính:** PyTorch, torchvision, transformers, timm, scipy, numpy, opencv-python, spaCy (cho scene graph parsing).
-   **GPU:** Khuyến nghị sử dụng GPU mạnh (ví dụ: A100 hoặc RTX 3090) cho quá trình huấn luyện đầy đủ.
