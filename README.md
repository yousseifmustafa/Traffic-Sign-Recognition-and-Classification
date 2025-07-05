```markdown
# 🚦 Traffic Sign Classification using CNN

This project is a deep learning-based Traffic Sign Classification system implemented in Python using Keras and TensorFlow. It leverages a Convolutional Neural Network (CNN) to accurately classify traffic signs into 43 different categories, making it suitable for autonomous driving applications, educational demonstrations, or as a base for further research.

---

## 📂 Project Structure

```

traffic-sign-classification/
│
├── traffic sign classification final.ipynb  # Jupyter Notebook with full code
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation (this file)
└── dataset/                                 # Directory with training and testing images (not included here)

```

---

## 🧠 Model Summary

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 30x30 RGB images
- **Classes:** 43 (e.g., Stop, Speed Limit, Yield, etc.)
- **Training Accuracy:** ~99%
- **Test Accuracy:** ~95%+

---

## 🛠️ Features

- Data loading and preprocessing (image resizing, normalization)
- CNN model architecture with Conv2D, MaxPooling2D, Dropout, and Dense layers
- Training with validation
- Evaluation on unseen test data
- Real-time prediction for custom images
- Performance visualization using plots

---

## 📊 Dataset

- **Source:** [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html)
- **Total Images:** Over 50,000 labeled images
- **Classes:** 43 different traffic signs

---

## 📥 Download Dataset

To run this notebook, you need to manually download the dataset from the official source:

👉 **[Download GTSRB Dataset](https://benchmark.ini.rub.de/Dataset_GTSRB.html)**

After downloading, extract and place the dataset in a folder named `dataset/` inside the project directory.

```

traffic-sign-classification/
└── dataset/
├── Train/
└── Test/

````

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yousseifmustafa/Traffic-Sign-Recognition-and-Classification.git
cd Traffic-Sign-Recognition-and-Classification
````

### 2. Install Requirements

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

> ✅ Compatible with Python 3.7+

### 3. Run the Notebook

```bash
jupyter notebook "traffic sign classification final.ipynb"
```

---

## 🧪 Example Prediction

After training the model, you can upload any traffic sign image, and the model will predict the corresponding class. The image is preprocessed and resized to 30x30 before prediction.

---

## 📈 Evaluation Metrics

* Accuracy
* Loss
* Confusion Matrix
* (Optional) Precision / Recall

---

## 📷 Visualization

The notebook includes visualizations of:

* Training vs Validation Accuracy
* Training vs Validation Loss

---

## ✅ Future Improvements

* Use data augmentation for better generalization
* Apply Transfer Learning (e.g., MobileNet, ResNet)
* Build a web or mobile app for real-time classification
* Add object detection (bounding box) support

---

**Made with ❤️ by [Yousseif Mustafa](https://github.com/yousseifmustafa)**

````
