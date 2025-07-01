# 🩺 Pneumonia Detection from Chest X-Rays using CNN

![GitHub repo size](https://img.shields.io/github/repo-size/your-username/pneumonia-detection-cnn)
![GitHub last commit](https://img.shields.io/github/last-commit/your-username/pneumonia-detection-cnn)
![GitHub stars](https://img.shields.io/github/stars/your-username/pneumonia-detection-cnn?style=social)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📌 Project Overview

This deep learning project uses a **Convolutional Neural Network (CNN)** to automatically detect **Pneumonia** from chest X-ray images. Built with **TensorFlow/Keras** and deployed with **Streamlit**, this model distinguishes between:
- ✅ `NORMAL` (Healthy lungs)
- ❌ `PNEUMONIA` (Infected lungs)

---

## 🎯 Features

- Upload a chest X-ray and get real-time prediction
- Clean and simple **Streamlit UI**
- CNN trained on **Kaggle Pneumonia dataset**
- Easy to run locally or deploy on the web

---


## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming |
| TensorFlow / Keras | Model building & training |
| Streamlit | Web app UI |
| Matplotlib | Accuracy/loss plotting |
| Git & GitHub | Version control |
| Git LFS | Handling large `.h5` model file |

---

# 🛠️ project structure

pneumonia_app/

├── app.py                  # 📱 Streamlit app for Pneumonia prediction

├── pneumonia_cnn_model.h5  # 🧠 Pre-trained CNN model (Keras)

├── requirements.txt        # 📦 Python dependencies

├── sample_images/          # 🖼️ Optional folder with sample chest X-ray images

└── README.md               # 📖 Project overview and setup instructions



---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn

```

### 2. install dependencies
```bash
pip install -r requirements.txt
```
### 3.Run the app
```bash
streamlit run app.py
```
---
### 🧠 How It Works
The CNN architecture:

- 3 Convolutional + MaxPooling layers

- Flatten → Dense → Dropout

- Output layer with sigmoid activation

- Trained with binary_crossentropy loss

---

### 🙏 Credits
- Dataset: Kaggle Chest X-ray Pneumonia Dataset

- Icons: Streamlit, Shields.io

- Developer: canishulk