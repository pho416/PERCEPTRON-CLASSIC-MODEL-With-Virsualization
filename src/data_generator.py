import pandas as pd
import numpy as np
import os

def generateDogCatDataset(num_samples=100, output_path='data/raw/DogOrCat.csv'):
    """
    Generates a dataset for dogs and cats and SAVES it to a CSV file.
    """
    num_cats = num_samples // 2
    num_dogs = num_samples - num_cats

    # --- Generate Features ---
    cat_heights = np.random.normal(loc=25, scale=5, size=num_cats)
    cat_weights = np.random.normal(loc=4.5, scale=1.5, size=num_cats)
    cat_barks = np.random.choice([0, 1], size=num_cats, p=[0.99, 0.01])
    cat_labels = np.zeros(num_cats)

    dog_heights = np.random.normal(loc=50, scale=15, size=num_dogs)
    dog_weights = np.random.normal(loc=15, scale=8, size=num_dogs)
    dog_barks = np.random.choice([0, 1], size=num_dogs, p=[0.05, 0.95])
    dog_labels = np.ones(num_dogs)

    # --- Combine and Shuffle ---
    all_heights = np.concatenate([cat_heights, dog_heights])
    all_weights = np.concatenate([cat_weights, dog_weights])
    all_barks = np.concatenate([cat_barks, dog_barks])
    all_labels = np.concatenate([cat_labels, dog_labels])

    X = np.stack([all_heights, all_weights, all_barks], axis=1)
    y = all_labels

    shuffled_indices = np.random.permutation(num_samples)
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    # --- THIS IS THE MISSING PART ---
    # 1. Create a Pandas DataFrame
    df = pd.DataFrame(X, columns=['chieu_cao', 'can_nang', 'co_sua'])
    df['loai'] = y

    # 2. Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3. Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    # --- END OF MISSING PART ---
    
    # We remove "return X, y" because the function's job is to create a file.