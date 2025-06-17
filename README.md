
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
