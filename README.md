# Kernel PCA for Dimensionality Reduction

This project implements Kernel Principal Component Analysis (Kernel PCA) for dimensionality reduction. The goal is to transform data into a lower-dimensional space while retaining its most important features. Kernel PCA is particularly useful for non-linearly separable data, making it an essential tool for complex datasets.

## Features

- **Kernel PCA Implementation**: Uses the RBF kernel for non-linear transformation of data.
- **Dataset Handling**: Loads and processes a dataset (`Wine.csv`) for dimensionality reduction.
- **Data Visualization**: Visualizes the results of dimensionality reduction.
- **Classification**: Applies a classifier (e.g., Logistic Regression) after dimensionality reduction to evaluate performance.

## Installation

To run this project, you need Python and the following libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. **Load the Dataset**: The dataset (`Wine.csv`) should be placed in the project directory.
2. **Run the Notebook**: Execute the cells in the Jupyter Notebook (`kernel_pca.ipynb`) to perform the following tasks:
   - Load and preprocess the data.
   - Apply Kernel PCA for dimensionality reduction.
   - Train and evaluate a classifier.

### Code Example

Here’s a quick example of how to load and preprocess the data:

```python
import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### Kernel PCA Implementation

The following code applies Kernel PCA:

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca = kpca.fit_transform(X)
```

## Results

After applying Kernel PCA, the reduced dataset is visualized using a scatter plot. This helps to observe the separation of classes in the reduced dimensions.

## Contributing

Contributions are welcome! Please follow the standard GitHub process:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.
