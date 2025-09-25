import numpy as np

class Perceptron:
    """
    Mô hình Perceptron cổ điển.

    Parameters
    ----------
    learning_rate : float
        Tốc độ học (giữa 0.0 và 1.0).
    n_iterations : int
        Số vòng lặp qua toàn bộ tập dữ liệu huấn luyện.

    Attributes
    ----------
    weights_ : 1d-array
        Các trọng số sau khi huấn luyện.
    bias_ : scalar
        Giá trị bias sau khi huấn luyện.
    """
    def __init__(self, learning_rate=0.01, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        """
        Huấn luyện mô hình với dữ liệu.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Dữ liệu huấn luyện.
        y : array-like, shape = [n_samples]
            Nhãn của dữ liệu.
        """
        n_samples, n_features = X.shape

        # 1. Khởi tạo trọng số và bias
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        # 2. Lặp qua toàn bộ dữ liệu n_iterations lần
        for _ in range(self.n_iterations):
            # 3. Lặp qua từng mẫu dữ liệu
            for idx, x_i in enumerate(X):
                # 4. Dự đoán nhãn
                prediction = self.predict(x_i)
                # 5. Tính toán giá trị cập nhật
                update = self.learning_rate * (y[idx] - prediction)
                # 6. Cập nhật trọng số và bias
                self.weights_ += update * x_i
                self.bias_ += update
        return self

    def net_input(self, X):
        """Tính toán tổng đầu vào có trọng số"""
        return np.dot(X, self.weights_) + self.bias_

    def predict(self, X):
        """Dự đoán nhãn (0 hoặc 1)"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)