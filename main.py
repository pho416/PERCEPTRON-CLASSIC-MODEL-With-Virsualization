import pandas as pd
import os  # Import the os module to check for file existence
from src.perceptron import Perceptron
from src.visualization import plot_decision_boundary
from src.data_generator import generateDogCatDataset # Import the generator function

# --- 1. Define data path and generate if needed ---
data_path = '../data/raw/DogOrCat.csv'

# Check if the data file exists. If not, create it.
if not os.path.exists(data_path):
    print(f"Data file not found at '{data_path}'.")
    print("Generating new dataset...")
    generateDogCatDataset(num_samples=300, output_path=data_path)
    print("-" * 20)

# --- 2. Load the data ---
print("Loading data...")
data_df = pd.read_csv(data_path)

# --- 3. Prepare the data for the model ---
# Separate features (X) and labels (y)
X = data_df[['chieu_cao', 'can_nang', 'co_sua']].values
y = data_df['loai'].values

# --- 4. Initialize and train the Perceptron model ---
print("Training the Perceptron model...")
model = Perceptron(learning_rate=0.01, n_iterations=100)
model.fit(X, y)
print("Training complete!")
print(f"Final Weights: {model.weights_}")
print(f"Final Bias: {model.bias_}")

# --- 5. Visualize the decision boundary ---
print("Plotting the decision boundary...")
plot_decision_boundary(X, y, model)