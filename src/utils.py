import numpy as np

def calculate_accuracy(y_true, y_pred):
    """
    Tính toán độ chính xác của mô hình.

    Parameters
    ----------
    y_true : array-like
        Các nhãn đúng (đáp án).
    y_pred : array-like
        Các nhãn được mô hình dự đoán.

    Returns
    -------
    float
        Độ chính xác, từ 0.0 đến 1.0.
    """
    accuracy = np.mean(y_true == y_pred)
    return accuracy