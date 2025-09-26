import pandas as pd
import os
from sklearn.preprocessing import StandardScaler # <<< THÊM DÒNG NÀY
from src.perceptron import Perceptron
from src.visualization import plot_decision_boundary
from src.data_generator import generateDogCatDataset
from src.utils import calculate_accuracy

data_path = '../data/raw/DogOrCat.csv'
if not os.path.exists(data_path):
    print(f"Data file not found. Generating new dataset...")
    generateDogCatDataset(num_samples=300, output_path=data_path)

data_df = pd.read_csv(data_path)
X = data_df[['chieu_cao', 'can_nang', 'co_sua']].values
y = data_df['loai'].values

print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
print("Features scaled successfully.")


print("\nTraining the Perceptron model on SCALED data...")
model = Perceptron(learning_rate=0.01, n_iterations=100)

model.fit(X_scaled, y) 
print("Training complete!")
print(f"Final Weights (after scaling): {model.weights_}")
print(f"Final Bias: {model.bias_}")

y_pred = model.predict(X_scaled)  # dùng dữ liệu đã scale
accuracy = calculate_accuracy(y, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# --- 5. Visualize the decision boundary ---
print("\nPlotting the decision boundary...")
# Truyền cả dữ liệu gốc (X) và scaler vào hàm
plot_decision_boundary(X, y, model, scaler) 