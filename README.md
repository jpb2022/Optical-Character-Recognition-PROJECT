

---

# 🧠 Optical Character Recognition (OCR) using Machine Learning

## 📌 Project Overview

This project focuses on building an end-to-end **Optical Character Recognition (OCR)** system using classical **machine learning and computer vision techniques**. The objective is to accurately extract printed characters from images and convert them into machine-readable text, enabling applications such as:

* 📄 Document digitization
* 🏦 Automated data entry
* 📚 Educational content processing
* 🧾 Form and invoice analysis

The project is designed and implemented by **Jitendra Kumar Gupta (M.Tech, IIT Kanpur)** as part of hands-on learning in **Machine Learning, Computer Vision, and Pattern Recognition**.

---

## 🔍 OCR Pipeline

The complete OCR workflow consists of the following stages:

1. **Image Acquisition**
   Load character images from the dataset.

2. **Image Preprocessing**
   Improve image quality and isolate text:

   * Grayscale conversion
   * Noise reduction (Gaussian / Median filtering)
   * Image thresholding (binary conversion)
   * Normalization and resizing

3. **Feature Extraction**
   Convert image pixels into feature vectors suitable for ML models:

   * Flattened pixel intensities
   * Edge-based structural features (optional)

4. **Model Training & Classification**
   Supervised learning models are trained to classify characters.

5. **Evaluation & Prediction**
   Performance is measured using accuracy and confusion matrices.

---

## 🤖 Machine Learning Models Implemented

This project compares multiple ML algorithms to evaluate OCR performance:

### ✅ Random Forest Classifier

* Ensemble of decision trees
* Robust to noise and overfitting
* Performs well on structured pixel features

### ✅ K-Nearest Neighbors (KNN)

* Distance-based classification
* Simple and interpretable
* Useful baseline for OCR tasks

### ✅ Support Vector Machine (SVM)

* Effective in high-dimensional spaces
* Kernel-based classification for complex boundaries
* Strong generalization capability

### ✅ Gaussian Mixture Model (GMM)

* Probabilistic clustering-based approach
* Useful for modeling character shape distributions
* Helps analyze class overlap and uncertainty

Each model is implemented and evaluated independently for comparison.

---

## 📊 Dataset

The project uses the **Standard OCR Dataset** from Kaggle:

🔗 Dataset Link:
[https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset)

### Dataset Highlights:

* Grayscale character images
* Multiple classes (alphabets and digits)
* Suitable for classical ML-based OCR pipelines

---

## 📁 Project Structure

```
Optical-Character-Recognition/
│
├── main.ipynb                # End-to-end OCR pipeline
├── Basic_EDA.ipynb           # Dataset exploration and visualization
│
├── Models/
│   ├── Random_Forest.ipynb
│   ├── KNN.ipynb
│   ├── SVM.ipynb
│   └── GMM.ipynb
│
├── Presentation/             # Project slides
├── Brief_Documentation.pdf   # Detailed technical report
└── README.md
```

---

## ⚙️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/jpb2022/Optical-Character-Recognition-PROJECT.git
cd Optical-Character-Recognition-PROJECT
```

### Step 2: Install Dependencies

Make sure Python 3.8+ is installed.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python
```

*(You may also use Anaconda for easier package management.)*

---

## ▶️ How to Run

1. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
2. Start with:

   * `Basic_EDA.ipynb` → Understand dataset & preprocessing
   * `main.ipynb` → Full OCR pipeline
3. Explore individual models inside the `Models/` folder.

---

## 📈 Evaluation Metrics

Model performance is evaluated using:

* ✔ Accuracy
* ✔ Confusion Matrix
* ✔ Class-wise Prediction Analysis

This allows comparison between different classifiers and helps identify misclassified characters.

---

## 🚀 Key Learning Outcomes

* Practical OCR pipeline development
* Image preprocessing for ML readiness
* Feature engineering from image data
* Comparative analysis of ML classifiers
* Understanding limitations of classical OCR vs Deep Learning

---

## 🔮 Future Improvements

Planned enhancements include:

* 🔹 CNN-based Deep Learning OCR using TensorFlow / PyTorch
* 🔹 Sequence-based recognition using CRNN + CTC Loss
* 🔹 Word-level text detection using OpenCV + EAST
* 🔹 Integration with real scanned documents
* 🔹 Deployment as a web-based OCR API

---

## 👨‍💻 Author

**Jitendra Kumar Gupta**
🎓 M.Tech — Industrial & Management Engineering (Data Science), IIT Kanpur
🎓 B.Tech — Mechanical Engineering, NIT Surat
💼 ML Engineer | Data Scientist | Educator

**Skills:**
Machine Learning, Deep Learning, Computer Vision, Python, SQL, Power BI, Generative AI, LLMs

📧 Email: [jitendraguptaaur@gmail.com](mailto:jitendraguptaaur@gmail.com)
🔗 GitHub: [https://github.com/jpb2022](https://github.com/jpb2022)

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---


