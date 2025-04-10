# Tuberculosis-Detection-Using-Deep-Learning

# 🫁 Tuberculosis Detection Using Deep Learning

## 📌 Project Overview
This project focuses on building a deep learning-based system to classify chest X-ray images as **Normal** or **Tuberculosis (TB)**. It leverages transfer learning with powerful CNN architectures and provides a Streamlit-based web interface for real-time predictions. The deployed model aims to assist healthcare professionals in early TB detection, especially in remote and underserved areas.

---

## 🚀 Demo
📍 **Live Application:** [Streamlit App on AWS](#)  
📁 **Dataset Source:** [Tuberculosis Chest X-rays Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-chest-xrays)

---

## 🧠 Skills Gained
- Python Scripting  
- Deep Learning with CNNs  
- Transfer Learning (ResNet50, VGG16, EfficientNetB0)  
- Image Preprocessing and Augmentation  
- Model Evaluation and Comparison  
- Streamlit Web Application Development  
- Deployment on AWS (EC2 / Elastic Beanstalk)

---

## 🏥 Domain
- Healthcare  
- Medical Imaging  
- Deep Learning

---

## 🧩 Problem Statement
Develop a deep learning-based solution to classify chest X-ray images as **Normal** or **TB-affected**, enabling:
- Early TB detection
- Automated screening in remote locations
- Decision support for radiologists
- Research on TB detection and prevalence

---

## 💼 Business Use Cases
- **Early Diagnosis**: Helps clinicians quickly identify TB symptoms from X-rays.
- **Rural Deployment**: Assists in diagnosis where radiologists are scarce.
- **Reduce Errors**: Serves as a second opinion for healthcare professionals.
- **Medical Research**: Supports TB trend analysis and academic research.

---

## 🛠️ Project Workflow

### 1. 🗂️ Data Preparation
- Dataset: 3008 images (TB: 2494 | Normal: 514)
- Split into train, validation, and test sets
- Handled corrupt/missing images
- Ensured class balance

### 2. 🧹 Preprocessing
- Image resizing & normalization
- Data augmentation (flip, rotate, zoom)
- EDA: Visualized class distribution and pixel stats

### 3. 🏗️ Model Development
- Transfer Learning with:
  - ResNet50  
  - VGG16  
  - EfficientNetB0  
- Fine-tuned layers and hyperparameters
- Added dropout & regularization

### 4. 📊 Evaluation
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC

### 5. 🌐 Application Development
- Built using **Streamlit**
- Users can upload X-ray images for real-time TB predictions
- Shows model confidence score

### 6. ☁️ Deployment
- Hosted on **AWS EC2**
- Scalable and accessible UI for real-world use

---

## 🖼️ Dataset Summary
- **Total Images**: 3008  
- **TB X-rays**: 2494  
- **Normal X-rays**: 514  
- Source: Kaggle ([Link](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-chest-xrays))

---

## 📈 Evaluation Metrics
| Model         | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------|----------|-----------|--------|----------|---------|
| ResNet50      | 93.1%    | 94.2%     | 91.8%  | 93.0%    | 0.96    |
| VGG16         | 91.0%    | 92.5%     | 89.3%  | 90.9%    | 0.94    |
| EfficientNetB0| 94.3%    | 95.1%     | 93.7%  | 94.4%    | 0.97    |

> 🏆 **EfficientNetB0** selected as the best-performing model.

---

## 🖥️ Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- AWS EC2 / Elastic Beanstalk
- Transfer Learning (CNNs)

---

## 📂 Folder Structure
