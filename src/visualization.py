import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

# The function signature is updated to accept 'scaler'
def plot_decision_boundary(X_original, y, model, scaler, title="Ranh giới quyết định của Perceptron"):
    """
    Vẽ ranh giới quyết định, xử lý dữ liệu đã được chuẩn hóa.
    
    Parameters:
    - X_original: Dữ liệu đặc trưng GỐC (chưa chuẩn hóa) để vẽ điểm.
    - y: Nhãn dữ liệu.
    - model: Mô hình Perceptron đã được huấn luyện.
    - scaler: Đối tượng StandardScaler đã được fit.
    """
    # Tách dữ liệu dựa trên đặc trưng thứ 3 (có sủa) từ dữ liệu GỐC
    barks_true = X_original[:, 2] == 1
    barks_false = X_original[:, 2] == 0

    # Tạo lưới điểm trên thang đo GỐC
    x_min, x_max = X_original[:, 0].min() - 5, X_original[:, 0].max() + 5
    y_min, y_max = X_original[:, 1].min() - 5, X_original[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Tạo các điểm lưới 3D trên thang đo GỐC
    third_feature_mean = X_original[:, 2].mean()
    mesh_original = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape[0], third_feature_mean)]
    
    # === THAY ĐỔI QUAN TRỌNG NHẤT ===
    # Chuyển đổi lưới điểm sang thang đo đã chuẩn hóa TRƯỚC KHI dự đoán
    mesh_scaled = scaler.transform(mesh_original)
    
    # Dự đoán trên lưới đã được chuẩn hóa
    Z = model.predict(mesh_scaled)
    Z = Z.reshape(xx.shape)

    # --- Phần vẽ đồ thị (giữ nguyên) ---
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Vẽ các điểm dữ liệu GỐC
    plt.scatter(X_original[barks_false, 0], X_original[barks_false, 1], c=y[barks_false], 
                cmap=plt.cm.coolwarm, edgecolor='k', marker='o', s=80)
    plt.scatter(X_original[barks_true, 0], X_original[barks_true, 1], c=y[barks_true], 
                cmap=plt.cm.coolwarm, edgecolor='k', marker='X', s=100)
    
    # Tạo chú thích
    cat_patch = mlines.Line2D([], [], color='#3b5897', marker='.', linestyle='None', markersize=15, label='Mèo (loai=0)')
    dog_patch = mlines.Line2D([], [], color='#a40000', marker='.', linestyle='None', markersize=15, label='Chó (loai=1)')
    no_bark_patch = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Không sủa')
    bark_patch = mlines.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=10, label='Có sủa')

    plt.legend(handles=[cat_patch, dog_patch, no_bark_patch, bark_patch], loc='lower right')
    
    plt.title(title)
    plt.xlabel("Chiều cao (cm)")
    plt.ylabel("Cân nặng (kg)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()