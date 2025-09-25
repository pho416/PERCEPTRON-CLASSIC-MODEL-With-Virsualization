import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

def plot_decision_boundary(X, y, model, title="Ranh giới quyết định của Perceptron"):
    """
    Vẽ ranh giới quyết định và hiển thị đặc trưng thứ 3 bằng ký hiệu (marker).
    """
    # Tách dữ liệu dựa trên đặc trưng thứ 3 (có sủa)
    # X[:, 2] == 1 là một mảng boolean, True nếu 'co_sua' là 1
    barks_true = X[:, 2] == 1
    barks_false = X[:, 2] == 0

    # Tạo một lưới điểm để vẽ vùng quyết định (như cũ)
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    third_feature_mean = X[:, 2].mean()
    mesh_3d = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape[0], third_feature_mean)]
    
    Z = model.predict(mesh_3d)
    Z = Z.reshape(xx.shape)

    # Bắt đầu vẽ
    plt.figure(figsize=(10, 7))
    
    # 1. Vẽ vùng quyết định
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # 2. Vẽ các điểm dữ liệu
    # Vẽ các điểm "không sủa" (marker='o')
    plt.scatter(X[barks_false, 0], X[barks_false, 1], c=y[barks_false], 
                cmap=plt.cm.coolwarm, edgecolor='k', marker='o', s=80,
                label='Không sủa')
    
    # Vẽ các điểm "có sủa" (marker='X')
    plt.scatter(X[barks_true, 0], X[barks_true, 1], c=y[barks_true], 
                cmap=plt.cm.coolwarm, edgecolor='k', marker='X', s=100,
                label='Có sủa')

    # 3. Tạo chú thích (legend) để giải thích các ký hiệu
    # Chú thích cho màu sắc (loài)
    cat_patch = mlines.Line2D([], [], color='#3b5897', marker='.', linestyle='None',
                              markersize=15, label='Mèo (loai=0)')
    dog_patch = mlines.Line2D([], [], color='#a40000', marker='.', linestyle='None',
                              markersize=15, label='Chó (loai=1)')
    # Chú thích cho ký hiệu (có sủa)
    no_bark_patch = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                  markersize=10, label='Không sủa (co_sua=0)')
    bark_patch = mlines.Line2D([], [], color='gray', marker='X', linestyle='None',
                               markersize=10, label='Có sủa (co_sua=1)')

    plt.legend(handles=[cat_patch, dog_patch, no_bark_patch, bark_patch], loc='lower right')
    
    plt.title(title)
    plt.xlabel("Chiều cao (cm)")
    plt.ylabel("Cân nặng (kg)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()