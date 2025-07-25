
# 📘 Internship Progress Overview

##  Week 1 Highlights

---

### Day 1–2: OCR Implementation & Answer Sheet Analysis (09/06/2025)

#### Task 1: OCR Using Python & Tesseract

**Objective**  
Implement Optical Character Recognition (OCR) to extract text from scanned documents.

**Tools & Libraries Used**  
- Python  
- OpenCV  
- Pillow  
- Tesseract OCR

**Process**  
- Preprocessed images: grayscale conversion, resizing, thresholding  
- Performed OCR using Tesseract  
- Exported extracted content to `.txt` files

---

####  Task 2: OCR Challenges in Academic Answer Sheets

**Sheet Types & Observations**
- **Theory-Based Sheets**: Standard OCR works; struggles with handwriting
- **Mathematical Sheets**: Struggles with symbols (e.g., √, ∑)
- **Diagram-Based Sheets**: OCR fails with diagrams; requires image analysis

**Challenges Encountered**
- Poor scan quality  
- Handwriting inconsistency  
- Mixed content (text + visuals)

**Key Takeaway**  
OCR effectiveness varies with content type. Accurate results require custom preprocessing and hybrid models tailored to academic formats.

---

### Day 3–5: Git & GitHub Overview (13/06/2025)

#### Objective  
Understand and apply version control using Git, and collaborate using GitHub.

#### 🛠️ Git Commands Practiced
- `git init`, `git add`, `git commit`, `git push`  
- `git status`, `git log --stat`  
- Branching and merging: `git checkout -b`, `git merge`

#### 🌐 GitHub Concepts Covered
- Cloud-based Git repository hosting  
- Pull requests, issues, and collaborative workflows

#### Activities Completed
- Created and pushed repositories  
- Tracked code changes with commits  
- Worked on files like `ocr.py`, `main.py`, and `web_scraper.py`  
- Analyzed commit history using `git log --stat`

#### Summary  
Acquired practical experience in source control and collaboration. Strengthened workflow discipline using Git and GitHub.

---

## Week 2 Highlights

---

### Day 6: Python Libraries for Data Science (16/06/2025)

#### 🎯 Objective
Explore and practice key Python libraries used in data science and machine learning workflows, focusing on matrix operations, sparse data structures, plotting, and data manipulation.

---

#### 🔧 Topics Covered

**1. NumPy**
- Created multi-dimensional arrays
- Performed array operations
- Built identity matrices using `np.eye()`

**2. SciPy**
- Converted NumPy arrays into sparse matrix formats:
  - `csr_matrix` (Compressed Sparse Row)
  - `coo_matrix` (Coordinate format)

**3. Matplotlib**
- Plotted mathematical functions (e.g., sine wave)
- Customized plots with markers using `%matplotlib inline`

**4. Pandas**
- Created and manipulated a DataFrame
- Filtered rows based on conditions (e.g., Age > 30)

---

#### Code Examples

```python
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])

from scipy import sparse
eye = np.eye(5)
sparse_matrix = sparse.csr_matrix(eye)

import matplotlib.pyplot as plt
x = np.linspace(-10,10,100)
y = np.sin(x)
plt.plot(x, y, marker="x")
plt.show()

import pandas as pd
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location': ["New York", "Paris", "Berlin", "London"],
        'Age': [24, 13, 53, 33]}
df = pd.DataFrame(data)
df[df.Age > 30]
```

---

#### 📌 Summary
Hands-on exposure to core libraries forming the foundation of data science:
- **NumPy** for numerical computation
- **SciPy** for scientific computing
- **Matplotlib** for data visualization
- **Pandas** for data analysis and filtering

---
##  DAY 7 – Supervised Learning: Datasets & k-NN Classifier(17/06/2025)

This notebook introduces the foundations of **supervised learning** with an emphasis on understanding datasets and implementing **k-Nearest Neighbors (k-NN) classification**. It includes:

###  Synthetic Datasets
- **Forge Dataset** – Used for binary classification with 2D visualization.
- **Wave Dataset** – Used for regression tasks with one feature.

###  Real-world Datasets
- **Breast Cancer Dataset** – Binary classification problem involving medical data.
- **Boston Housing Dataset** – Regression dataset with extended features.

###  Machine Learning Concepts
- Splitting datasets using `train_test_split`
- Training a `KNeighborsClassifier`
- Visualizing decision boundaries for different `k` values
- Plotting accuracy vs. `k` to evaluate overfitting vs. underfitting

---

##  DAY 8 – Supervised Learning: k-NN Regression(18/06/2025)

This notebook extends the understanding of **k-NN into regression tasks**. It provides a visual and intuitive explanation of how changing the number of neighbors impacts regression predictions.

###  Core Topics
- `KNeighborsRegressor` – k-Nearest Neighbors Regression
- Effect of changing the `n_neighbors` parameter
- Visualizing smoothness vs. complexity in regression curves
- Practical use of `mglearn` to generate examples
##  DAY 9 – Supervised Learning: Linear Models for Classification(19/06/2025)

This notebook explores the application of **linear models** in both **binary** and **multiclass classification** using `LogisticRegression` and `LinearSVC`. It focuses on:

- Understanding the **linear decision boundary equation**
- Visualizing classification boundaries using the **Forge** dataset
- Evaluating the impact of **regularization (C)** on model performance
- Comparing **L1 and L2 regularization** effects on model coefficients
- Handling real-world data (**Breast Cancer dataset**) with feature scaling
- Performing **multiclass classification** using synthetic `make_blobs` data and interpreting multiple linear decision line
  -----
## DAY 10 Summary – Supervised Learning with scikit-learn

###  Decision Trees
- Trained and visualized decision trees.
- Explored model accuracy and feature importances on the Breast Cancer dataset.

###  Tree-Based Regression
- Compared predictions of `DecisionTreeRegressor` and `LinearRegression` on RAM price data.
- Used log-scale plotting for better visualization.

###  Random Forests
- Built multiple decision trees using `RandomForestClassifier`.
- Visualized individual trees’ decision boundaries.
- Evaluated ensemble performance.

###  Gradient Boosting
- Used `GradientBoostingClassifier` with various hyperparameters.
- Analyzed accuracy and feature importances.

###  Support Vector Machines (SVM)
- Trained both linear and kernel-based SVMs.
- Demonstrated decision boundaries in 2D and 3D spaces.
- Explored the effects of parameters `C` and `gamma`.

###  Neural Networks (MLP)
- Trained `MLPClassifier` on synthetic (`make_moons`) and real-world datasets.
- Tuned hidden layers, solver, `alpha` (regularization), and performed input scaling.
- Visualized weight matrix to understand learned features.

---

> **Skills Gained**: Training & tuning models, data scaling, decision surface visualization, performance evaluation, and using scikit-learn’s ensemble and neural network models effectively.

## Week 3 Highlights
## DAY 11-12 Summary – Unsupervised Learning with scikit-learn
### What I Learned

- How to use **PCA** to reduce dimensions of image data  
- How to apply clustering algorithms:
  - **KMeans**
  - **DBSCAN**
  - **Agglomerative (Hierarchical) Clustering**
- How to visualize:
  - **Principal components** (eigenfaces)
  - **Clusters and noise**
  - **Dendrograms** for hierarchical relationships
- How to evaluate clustering with **Adjusted Rand Index (ARI)**

---

### Tools & Libraries Used

- **Python**
- `scikit-learn`, `matplotlib`, `scipy`, `mglearn`

---

### Outcome

I successfully visualized clusters of similar faces, identified outliers, and understood how different clustering methods behave on high-dimensional data after applying PCA. These steps helped strengthen my understanding of **unsupervised learning** and its application to real-world datasets.

## DAY 13-14 Summary – Unsupervised Learning with scikit-learn

## 🔍 What I Learned

- How to **classify faces** using:
  - **k-Nearest Neighbors (k-NN)**
- How to apply **unsupervised clustering** to explore the dataset
- How to evaluate model accuracy using:
  - **Confusion Matrix**
  - **Accuracy Score**
- How to visualize:
  - PCA-transformed data
  - Clustered faces and noise
- Basics of feature scaling and dimensionality reduction

---
#  Day 15 – CHAPTER 3 -Feature Representation & Categorical Encoding

This notebook explores how to represent data effectively for machine learning models, focusing on categorical variables and feature engineering.

- Differentiated **continuous** vs **categorical** features using the Adult Income dataset.
- Applied **one-hot encoding** with `pandas.get_dummies()` to convert categorical data into numerical format.
- Highlighted risks of encoding training and test sets separately, which can break model compatibility.
- Demonstrated how **numeric categorical features** must be explicitly treated using `.astype(str)`.
- Trained a **logistic regression** model and resolved convergence issues with `max_iter`.
- Compared behavior of **linear models** and **decision trees** using a synthetic dataset.
- Emphasized the importance of correct feature representation in model performance.

