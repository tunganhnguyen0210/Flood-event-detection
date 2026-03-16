# Flood-event-detection

# 🌊 Flood Image Detection from Social Media

> **Course:** DPL302m — Deep Learning  
> **Task:** Multimedia Satellite Task — Flood Evidence Retrieval  
> **Platform:** Kaggle (NVIDIA Tesla T4 GPU)

---

## 📋 Mô tả dự án

Dự án này xây dựng hệ thống tự động phát hiện hình ảnh chứa bằng chứng về sự kiện **ngập lụt** từ các luồng dữ liệu mạng xã hội (social media). Hệ thống sử dụng phương pháp **đa phương thức (multimodal)**, kết hợp cả **dữ liệu văn bản (metadata)** và **dữ liệu hình ảnh** để phân loại nhị phân: ảnh có chứa bằng chứng ngập lụt hay không.

### Định nghĩa bài toán
- **True Positive:** Hình ảnh thể hiện mức nước cao bất thường trong các khu vực dân cư, công nghiệp, thương mại hoặc nông nghiệp.
- **Thách thức chính:**
  - Phân biệt mực nước bình thường (hồ, sông) với mực nước ngập bất thường (đường phố, nhà cửa).
  - Xử lý nhiều loại ngập lụt khác nhau: ngập ven biển, ngập sông, ngập do mưa lớn.

---

## 🏗️ Kiến trúc hệ thống

Hệ thống gồm **2 nhánh (branch)** chính:

### 1. Nhánh Văn bản (Text Branch) — LSTM Classifier

Sử dụng metadata (title, description, user_tags) để phân loại.

```
Input (user_tags) → Embedding → LSTM (2 layers) → LayerNorm → Dropout → FC → ReLU → FC → Output
```

| Thành phần       | Chi tiết                          |
|-------------------|-----------------------------------|
| **Embedding**     | vocab_size ≈ 13,531, dim = 64    |
| **LSTM**          | hidden_size = 64, 2 layers       |
| **Normalization** | LayerNorm                         |
| **Dropout**       | 0.3                               |
| **Fully Connected** | 64 → 16 → 2 (classes)         |
| **Optimizer**     | Adam, lr = 1e-4                   |
| **Scheduler**     | ReduceLROnPlateau (patience=5)    |
| **Epochs**        | 50                                |

### 2. Nhánh Hình ảnh (Image Branch) — CNN Classifier

Sử dụng trực tiếp ảnh từ tập devset để phân loại.

| Thành phần          | Chi tiết                                |
|----------------------|-----------------------------------------|
| **Input Size**       | 224 × 224 × 3                          |
| **Normalization**    | ImageNet mean/std                       |
| **Data Augmentation**| RandomFlip, ColorJitter, RandomRotation, RandomResizedCrop, RandomErasing |
| **Batch Size**       | 32                                      |

---

## 📁 Cấu trúc dữ liệu

```
📦 Dataset (Kaggle: 2023sumdpl302m)
├── devset_images/              # Thư mục ảnh (development set)
│   └── devset_images/          # Các file ảnh (image_id.jpg)
├── devset_images_gt.csv        # Ground truth labels (id, label: 0/1)
├── devset_images_metadata.json # Metadata: title, description, user_tags
└── test.csv                    # Test set metadata: title, description, user_tags, image_id
```

### Cấu trúc dữ liệu metadata

| Cột           | Mô tả                                      |
|----------------|---------------------------------------------|
| `id`           | ID duy nhất của ảnh (int64)                |
| `title`        | Tiêu đề ảnh trên mạng xã hội              |
| `description`  | Mô tả chi tiết do người dùng cung cấp     |
| `user_tags`    | Các tag do người dùng gắn                  |
| `label`        | 0 = Không ngập lụt, 1 = Có ngập lụt       |

---

## ⚙️ Pipeline xử lý

### Tiền xử lý văn bản (Text Preprocessing)
1. Chuyển về chữ thường (lowercase)
2. Loại bỏ dấu Unicode (`unidecode`)
3. Loại bỏ ký tự đặc biệt (chỉ giữ alphanumeric)
4. Loại bỏ stopwords (English)
5. Stemming (Porter Stemmer)
6. Xây dựng vocabulary từ cả dev set và test set
7. Tokenization + Padding (max_seq_len = 32/50)

### Tiền xử lý ảnh (Image Preprocessing)
1. Resize về 224 × 224
2. Chuẩn hóa theo ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. Data Augmentation:
   - Random Horizontal/Vertical Flip
   - Color Jitter (brightness, contrast, saturation, hue)
   - Random Rotation (±15°)
   - Random Resized Crop
   - Random Erasing (p=0.75)

---

## 📊 Kết quả

### Text Model (LSTM trên user_tags)

| Metric          | Giá trị     |
|-----------------|-------------|
| Train Accuracy  | **87.38%**  |
| Val Accuracy    | **83.90%**  |
| Train Loss      | 0.3069      |
| Val Loss        | 0.3846      |

> Mô hình bắt đầu hội tụ từ epoch ~23, đạt hiệu suất tốt nhất quanh epoch 41-43.

---

## 🚀 Hướng dẫn sử dụng

### Yêu cầu hệ thống
- Python 3.10+
- GPU NVIDIA (khuyến nghị)

### Cài đặt thư viện

```bash
pip install torch torchvision pandas numpy matplotlib opencv-python Pillow nltk unidecode scikit-learn
```

### Chạy notebook

1. Upload notebook `dpl302m.ipynb` lên Kaggle
2. Thêm dataset `2023sumdpl302m` vào notebook
3. Bật GPU accelerator (Tesla T4)
4. Chạy tuần tự các cell

### Kết quả đầu ra

File `test_with_labels.csv` chứa kết quả dự đoán cho test set:

```csv
id,label
3483809003,1
3712805295,0
...
```

---

## 🛠️ Công nghệ sử dụng

| Thư viện         | Mục đích                          |
|-------------------|-----------------------------------|
| PyTorch           | Framework deep learning chính     |
| torchvision       | Xử lý ảnh và augmentation         |
| NLTK              | Xử lý ngôn ngữ tự nhiên          |
| scikit-learn      | Train/test split                   |
| OpenCV / Pillow   | Đọc và xử lý ảnh                  |
| pandas / numpy    | Xử lý dữ liệu tabular            |
| matplotlib        | Trực quan hóa kết quả              |
| unidecode         | Chuẩn hóa Unicode                  |

---

## 📝 Ghi chú

- Mô hình text sử dụng **user_tags** làm đặc trưng chính, cho kết quả tốt vì tags thường chứa từ khóa liên quan trực tiếp đến sự kiện (flood, storm, water, etc.)
- Phần Image model đang trong giai đoạn phát triển, sử dụng cùng bộ dữ liệu devset cho cả train và validation (cần cải thiện bằng cách split riêng biệt)
- Có thể cải thiện thêm bằng cách:
  - Sử dụng pretrained CNN (ResNet, EfficientNet) cho nhánh ảnh
  - Kết hợp (fusion) kết quả từ cả 2 nhánh text và image
  - Áp dụng cross-validation
  - Sử dụng title và description bên cạnh user_tags

---

## 📄 Tham khảo

- [MediaEval Multimedia Satellite Task](http://www.multimediaeval.org/)
- [YFCC100M Dataset](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/)
- PyTorch Documentation: https://pytorch.org/docs/
- NLTK Documentation: https://www.nltk.org/


