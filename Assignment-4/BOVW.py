import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class_1_path = "class_1_images-20241208"
class_2_path = "class_2_images-20241208"

patch_size = 8
n_clusters = 10

# Function to extract 8x8 patches and flatten to 192-dimensional vectors
def extractPatches(image, patch_size):
    h, w, c = image.shape
    patches = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size].flatten()
            patches.append(patch)
    return np.array(patches)

# Load images and extract patches
def loadPatches(folder_path):
    patches = []
    for file in sorted(os.listdir(folder_path)):
        image = cv2.imread(os.path.join(folder_path, file))  # imread reads the image into Numpy array (H,W,3) in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_patches = extractPatches(image, patch_size)
        patches.extend(image_patches)
    return np.array(patches)

# Load patches from both folders
patches_class1 = loadPatches(class_1_path)
patches_class2 = loadPatches(class_2_path)
all_patches = np.vstack((patches_class1, patches_class2))

# Perform k-means clustering to get cluster centers
kmeans = KMeans(n_clusters=n_clusters, random_state=40)
kmeans.fit(all_patches)
cluster_centers = kmeans.cluster_centers_

# Compute histogram for a single image
def compute_histogram(image_path, cluster_centers):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    patches = extractPatches(image, patch_size)
    histogram = np.zeros(len(cluster_centers))
    for patch in patches:
        distances = np.linalg.norm(cluster_centers - patch, axis=1)
        closest_cluster = np.argmin(distances)
        histogram[closest_cluster] += 1
    return histogram

# Compute histograms for all images in a folder
def compute_histograms(folder_path, cluster_centers):
    histograms = []
    for file in sorted(os.listdir(folder_path)):
        histogram = compute_histogram(os.path.join(folder_path, file), cluster_centers)
        histograms.append(histogram)
    return np.array(histograms)

histograms_class_1 = compute_histograms(class_1_path, cluster_centers)
histograms_class_2 = compute_histograms(class_2_path, cluster_centers)

fig, axes = plt.subplots(2, 5, figsize=(20, 10))

for i, histogram in enumerate(histograms_class_1):
    axes[0, i].bar(range(n_clusters), histogram, color='blue', alpha=0.7)
    axes[0, i].set_title(f"Class 1 - Image {i+1}")
    axes[0, i].set_xlabel("Cluster")
    axes[0, i].set_ylabel("Frequency")

for i, histogram in enumerate(histograms_class_2):
    axes[1, i].bar(range(n_clusters), histogram, color='green', alpha=0.7)
    axes[1, i].set_title(f"Class 2 - Image {i+1}")
    axes[1, i].set_xlabel("Cluster")
    axes[1, i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()