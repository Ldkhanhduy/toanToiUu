# Đồ Án Học Phần Toán Tối Ưu: Hồi Quy Phân Vị & Các Thuật Toán Tối Ưu Tự Triển Khai

Dự án này tập trung vào việc nghiên cứu và triển khai mô hình **Hồi quy Phân vị (Quantile Regression)** cùng với các thuật toán tối ưu hóa nền tảng (handmade) từ thư viện Numpy để giải quyết bài toán dự báo thời gian giao hàng (`Time_taken`) trên tập dữ liệu Food Delivery (`foodDeli`).

---

## 1. Tổng Quan Bài Toán & Cơ Sở Toán Học

### 1.1. Hồi quy Phân vị (Quantile Regression)
Trong khi hồi quy tuyến tính thông thường (OLS) tối thiểu hóa sai số bình phương trung bình (MSE) để dự báo giá trị trung bình (mean), hồi quy phân vị cho phép ước lượng các phân vị khác nhau (quantiles) của phân phối điều kiện. Điều này cực kỳ hữu ích đối với các bài toán có dữ liệu không phân phối chuẩn, có nhiều nhiễu hoặc khi chúng ta đặc biệt quan tâm đến các trường hợp biên (ví dụ: phân vị thứ 90 để đảm bảo giao hàng đúng giờ trong 90% trường hợp).

### 1.2. Hàm mất mát Pinball Loss
Để ước lượng phân vị thứ $\tau \in (0, 1)$, chúng ta sử dụng hàm mất mát **Pinball Loss** (hay còn gọi là tilted absolute loss):

$$L_{\tau}(y, \hat{y}) = \frac{1}{m} \sum_{i=1}^{m} \rho_{\tau}(y_i - \hat{y}_i)$$

Trong đó hàm penalty $\rho_{\tau}(u)$ được định nghĩa là:

$$\rho_{\tau}(u) = u \cdot (\tau - \mathbb{I}_{\{u < 0\}}) = \begin{cases} \tau \cdot u & \text{nếu } u \geq 0 \\ (\tau - 1) \cdot u & \text{nếu } u < 0 \end{cases}$$

Với bài toán này, mô hình được cấu hình để tối ưu hóa tại phân vị **$\tau = 0.9$** nhằm mục đích dự báo ngưỡng thời gian giao hàng tối đa mà 90% đơn hàng sẽ hoàn thành trước thời điểm đó.

### 1.3. Tính Subgradient
Hàm Pinball Loss không khả vi tại điểm sai số bằng $0$. Do đó, để tối ưu hóa trọng số $\theta$, ta sử dụng phương pháp **Subgradient Descent** (Giảm dưới gradient).
Dưới gradient của hàm mất mát đối với vector trọng số $\theta$ được tính như sau:

$$\nabla_{\theta} L_{\tau} = \frac{1}{m} X^T p$$

Trong đó vector trung gian $p \in \mathbb{R}^m$ có phần tử thứ $i$ xác định bởi:

$$p_i = \begin{cases} -\tau & \text{nếu } y_i - \hat{y}_i \geq 0 \\ 1 - \tau & \text{nếu } y_i - \hat{y}_i < 0 \end{cases}$$

---

## 2. Cấu Trúc Dự Án

Thư mục dự án được tổ chức như sau:

```text
├── dataset/                     # Thư mục chứa dữ liệu
│   ├── foodDeli/                # Dữ liệu gốc (chưa xử lý)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── Sample_Submission.csv
│   └── foodDeli_processed/      # Dữ liệu đã được tiền xử lý sạch
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── notebooks/                   # Jupyter Notebooks phục vụ phân tích & huấn luyện
│   ├── read_data.ipynb          # Đọc và trực quan hóa dữ liệu sơ bộ
│   ├── data_processing.ipynb    # Tiền xử lý dữ liệu (chuẩn hóa, encode, drop na,...)
│   └── training/                # Lưu trữ quá trình train chi tiết của từng thuật toán
│       ├── Adam.ipynb
│       ├── Nesterov.ipynb
│       ├── RMSProp.ipynb
│       └── useMomentumTrain.ipynb
│
├── src/                         # Mã nguồn triển khai mô hình và thuật toán tối ưu
│   ├── models/
│   │   └── quantile_regression.py   # Lớp mô hình QuantileRegressionV1
│   └── optimizers/
│       ├── adam.py                  # Thuật toán Adam
│       ├── MomentumOptimizerHandMade.py  # Thuật toán GD với Momentum
│       ├── Nesterov.py              # Thuật toán Nesterov Accelerated Gradient (NAG)
│       └── RMSProp.py               # Thuật toán RMSProp
│
├── results/                     # Kết quả đầu ra sau khi chạy huấn luyện
│   ├── weights/                 # Lưu file trọng số (.npy) và lịch sử loss của các thuật toán
│   └── plots/                   # Biểu đồ so sánh tốc độ hội tụ (loss curve)
│
├── main.py                      # Script đánh giá hiệu năng và so sánh kết quả các thuật toán
├── requirements.txt             # Danh sách các thư viện cần cài đặt
└── README.md                    # Hướng dẫn dự án (File này)
```

---

## 3. Các Thuật Toán Tối Ưu Tự Viết (Handmade Optimizers)

Dự án tự xây dựng 4 thuật toán tối ưu hóa phổ biến dựa trên Numpy:

1. **Gradient Descent kết hợp Momentum (`MomentumOptimizerHandMade.py`)**:
   Tích lũy các vector cập nhật từ các bước trước để vượt qua các vùng yên ngựa (saddle points) và giảm dao động:
   $$v_t = \gamma v_{t-1} - \eta \nabla_{\theta} L$$
   $$\theta_t = \theta_{t-1} + v_t$$

2. **Nesterov Accelerated Gradient (NAG) (`Nesterov.py`)**:
   Dự đoán vị trí tiếp theo của tham số trước khi thực hiện cập nhật nhằm điều chỉnh hướng đi thông minh hơn:
   $$v_t = \gamma v_{t-1} + \nabla_{\theta} L$$
   $$step = \nabla_{\theta} L + \gamma v_t$$
   $$\theta_t = \theta_{t-1} - \eta \cdot step$$

3. **RMSProp (`RMSProp.py`)**:
   Tự động điều chỉnh tốc độ học cho từng tham số dựa trên trung bình trượt của bình phương gradient gần đây, giúp kiểm soát tốt tốc độ học trong không gian có độ dốc thay đổi mạnh:
   $$v_t = \beta v_{t-1} + (1 - \beta) (\nabla_{\theta} L)^2$$
   $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} \nabla_{\theta} L$$

4. **Adam (`adam.py`)**:
   Sự kết hợp giữa Momentum (ước lượng moment bậc 1) và RMSProp (ước lượng moment bậc 2) kèm theo hiệu chỉnh độ lệch (bias correction) ở các bước đầu:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{\theta} L$$
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_{\theta} L)^2$$
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 4. Hướng Dẫn Cài Đặt và Sử Dụng

### Bước 1: Khởi tạo và Kích hoạt Môi trường ảo (Khuyến nghị)
Mở terminal tại thư mục gốc của dự án:
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows Powershell)
.\venv\Scripts\Activate.ps1

# Kích hoạt môi trường ảo (Linux / MacOS)
source venv/bin/activate
```

### Bước 2: Cài đặt thư viện yêu cầu
Cài đặt các gói phụ thuộc cần thiết bằng cách chạy:
```bash
pip install -r requirements.txt
```

### Bước 3: Tiền xử lý dữ liệu và Huấn luyện mô hình
Các bước tiền xử lý dữ liệu và huấn luyện mô hình được chi tiết hóa trong các file Jupyter Notebook tại thư mục `notebooks/`. Bạn có thể mở các notebook này lên và chạy tuần tự để tạo ra các file trọng số `.npy` lưu trong `results/weights/`.

### Bước 4: Chạy Đánh giá và So sánh (main.py)
Để chạy so sánh hiệu năng giữa các thuật toán tối ưu tự viết và mô hình chuẩn `QuantileRegressor` của thư viện Scikit-learn, thực hiện lệnh:
```bash
python main.py
```
*Lưu ý: Nếu gặp lỗi hiển thị ký tự Unicode tiếng Việt trên terminal Windows, hãy chạy lệnh dưới đây:*
```bash
python -X utf8 main.py
```

---

## 5. Kết Quả Thực Nghiệm & Bảng Xếp Hạng

Sau khi huấn luyện các mô hình tại phân vị $\tau = 0.9$ trên dữ liệu huấn luyện đã xử lý (`train_processed.csv`), chúng được đánh giá trên tập dữ liệu kiểm thử độc lập (`test_processed.csv`). 

Kết quả thu được như sau:

| Bảng xếp hạng | Thuật toán | Pinball Loss ($\tau = 0.9$) | MAE | RMSE |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Momentum (Hand-made)** | **1.4116** | **4.1130** | **6.9405** |
| **2** | RMSProp (Hand-made) | 1.4589 | 4.4754 | 7.1264 |
| **3** | Adam (Hand-made) | 1.4646 | 4.4841 | 7.1147 |
| **4** | Sklearn (QuantileRegressor) | 1.4880 | 4.7077 | 7.2415 |
| **5** | Nesterov (Hand-made) | 1.4939 | 4.3411 | 7.2492 |

### Nhận xét:
* Thuật toán **Momentum (Hand-made)** cho hiệu năng tối ưu tốt nhất trên tập dữ liệu kiểm thử này, đạt giá trị Pinball Loss thấp nhất (1.4116) cùng chỉ số MAE thấp nhất (4.1130).
* Các thuật toán tự triển khai như **Momentum**, **RMSProp** và **Adam** đều cho kết quả Pinball Loss vượt trội (thấp hơn) so với mô hình tối ưu mặc định `QuantileRegressor` của thư viện **Sklearn**.
* Biểu đồ đường suy giảm hàm lỗi (loss curve) của từng thuật toán được lưu tự động tại thư mục `results/plots/best_optimizers_comparison.png` để trực quan hóa quá trình hội tụ trong suốt các epoch huấn luyện.