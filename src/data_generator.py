import pandas as pd
import numpy as np
import os

def generateDogCatDataset(num_samples=300, output_path='../data/raw/DogOrCat.csv'):
    """
    Tạo và lưu một bộ dữ liệu ngẫu nhiên về chó và mèo.

    Bộ dữ liệu bao gồm 3 đặc trưng (chiều cao, cân nặng, có sủa) và 1 nhãn
    (0 cho mèo, 1 cho chó). File sẽ được lưu dưới dạng CSV.

    Args:
        num_samples (int): Tổng số lượng mẫu dữ liệu cần tạo.
        output_path (str): Đường dẫn để lưu file CSV.
    """
    # Chia đều số lượng mẫu cho chó và mèo
    num_cats = num_samples // 2
    num_dogs = num_samples - num_cats

    # Tạo dữ liệu cho Mèo (nhãn = 0)
    cat_features = {
        'height': np.random.normal(loc=25, scale=5, size=num_cats),
        'weight': np.random.normal(loc=4.5, scale=1.5, size=num_cats),
        'barks': np.random.choice([0, 1], size=num_cats, p=[0.99, 0.01])
    }
    cat_labels = np.zeros(num_cats)

    # Tạo dữ liệu cho Chó (nhãn = 1)
    dog_features = {
        'height': np.random.normal(loc=55, scale=12, size=num_dogs),
        'weight': np.random.normal(loc=20, scale=8, size=num_dogs),
        'barks': np.random.choice([0, 1], size=num_dogs, p=[0.05, 0.95])
    }
    dog_labels = np.ones(num_dogs)

    # Kết hợp và xáo trộn dữ liệu
    all_features = np.concatenate([
        np.stack(list(cat_features.values()), axis=1),
        np.stack(list(dog_features.values()), axis=1)
    ])
    all_labels = np.concatenate([cat_labels, dog_labels])

    shuffled_indices = np.random.permutation(num_samples)
    X = all_features[shuffled_indices]
    y = all_labels[shuffled_indices]

    # Tạo DataFrame và lưu ra file CSV
    df = pd.DataFrame(X, columns=['chieu_cao', 'can_nang', 'co_sua'])
    df['loai'] = y

    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset with {num_samples} samples was successfully generated at '{output_path}'.")

# Thêm khối này để có thể chạy file trực tiếp và kiểm thử