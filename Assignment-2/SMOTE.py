import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def SMOTE(X, y, k=5):

    # Identify the minority class and its size
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
 
    majorityClass = max(class_counts, key=class_counts.get)
    minorityClass = min(class_counts, key=class_counts.get)
    n_minority_samples = class_counts[minorityClass]
    n_majority_samples = class_counts[majorityClass]
    
    imbalance = n_majority_samples - n_minority_samples  # Calculate how many synthetic samples to generate

    # Get the minority class instances
    minority_indices = np.where(y == minorityClass)[0]
    minority_samples = X[minority_indices]

    neighbors = NearestNeighbors(n_neighbors=k).fit(minority_samples) # k nearest neighbors for each minority sample
    
    synthetic_samples = []
    
    for i in range(len(minority_samples)):
        neighbor_indices = neighbors.kneighbors(minority_samples[i].reshape(1, -1), return_distance=False)[0]  # Find neighbors for each minority sample
        
        # Generate synthetic samples
        for j in range(imbalance // len(minority_samples)):
            random_neighbor_index = np.random.choice(neighbor_indices[1:], 1)[0]  # Exclude itself
            diff = minority_samples[random_neighbor_index] - minority_samples[i]
            gap = np.random.rand()  # Random gap between 0 and 1
            
            synthetic_sample = minority_samples[i] + gap * diff
            synthetic_samples.append(synthetic_sample)

    # Combine original and synthetic samples
    X_resampled = np.vstack((X, synthetic_samples))
    y_resampled = np.hstack((y, [minorityClass] * len(synthetic_samples)))

    return X_resampled, y_resampled


class_1 = np.load('class1_data.npy')  # class_1.npy contains 1000 points
class_2 = np.load('class2_data.npy')  # class_2.npy contains 100 points

# Combine the classes into a single dataset
X = np.vstack((class_1, class_2))
y = np.array([0] * len(class_1) + [1] * len(class_2))  # 0 for class_1, 1 for class_2

X_resampled, y_resampled = SMOTE(X, y, k=5)

# Separate the synthetic data points for visualization
class_2_synthetic = X_resampled[y_resampled == 1]
class_2_original = class_2

plt.figure(figsize=(10, 6))
plt.scatter(class_1[:, 0], class_1[:, 1], label='Class 1 (Original)', alpha=0.5, color='blue')
plt.scatter(class_2_original[:, 0], class_2_original[:, 1], label='Class 2 (Original)', color='red')
plt.scatter(class_2_synthetic[:, 0], class_2_synthetic[:, 1], label='Class 2 (Synthetic)', alpha=0.4, color='yellow')

plt.title('Comparison of Original and Synthetic Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()