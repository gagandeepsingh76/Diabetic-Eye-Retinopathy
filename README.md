# 🧠 Diabetic Eye Retinopathy Detection

This repository contains multiple deep learning experiments for **Diabetic Eye Retinopathy (DR) classification** using state-of-the-art Convolutional Neural Network (CNN) architectures. The goal is to develop automated systems to detect DR from retinal images with high accuracy, exploring various models, optimizers, and training strategies.

---

## 📌 Overview

Diabetic Retinopathy is a diabetes-related complication that affects the eyes and can lead to blindness if untreated. Early detection is critical, and automated DR classification can assist ophthalmologists by providing fast and reliable screening.

This project implements and compares several CNN models on a **dataset of 1,500 retinal images** divided into **2 classes** (DR and No DR). Model accuracies range from **70% to 85%**, depending on architecture and training strategies.

---

## 📂 Project Structure

Diabetic-Eye-Retinopathy

- MobileNetV3.ipynb # MobileNetV3 model for lightweight DR detection
- efficientnetB0.ipynb # EfficientNet-B0 baseline
- inceptionv3.ipynb # InceptionV3 baseline
- inceptionv3__.ipynb # InceptionV3 variation with minor tweaks
- inceptionv3_enh.ipynb # Enhanced InceptionV3 with advanced training strategies
- resnet152ADAM.ipynb # ResNet152 with Adam optimizer
- resnet152ADAM1.ipynb # ResNet152 with modified Adam optimizer
- resnet152ADAMAX.ipynb # ResNet152 with Adamax optimizer
- resnet152NADAM.ipynb # ResNet152 with Nadam optimizer
- resnet152RMSPROP.ipynb # ResNet152 with RMSprop optimizer


---

## 🧩 Dataset

- **Size:** 1,500 retinal images  
- **Classes:** 2 (DR, No DR)  
- **Preprocessing:**
  - Resizing to match input shape of models
  - Normalization
  - Data augmentation (rotation, flipping, zoom)  

---

## 📌 Implemented Models

| Model | Key Features | Accuracy Range |
|-------|--------------|----------------|
| InceptionV3 | Baseline CNN | 72-78% |
| InceptionV3 Variation | Tweaked layers and training | 73-79% |
| Enhanced InceptionV3 | Advanced training strategies, fine-tuning | 78-83% |
| ResNet152 + Adam | Deep residual network | 75-82% |
| ResNet152 + Modified Adam | Optimized Adam variants | 76-83% |
| ResNet152 + Adamax | Adamax optimizer | 74-81% |
| ResNet152 + Nadam | Nadam optimizer | 75-84% |
| ResNet152 + RMSprop | RMSprop optimizer | 73-80% |
| EfficientNet-B0 | Lightweight and efficient | 70-78% |
| MobileNetV3 | Mobile-friendly, fast inference | 70-77% |

---

## 🛠️ Tech Stack

- **Language:** Python 🐍  
- **Deep Learning:** TensorFlow / Keras  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Model Evaluation:** Scikit-learn  

---

## 📊 Performance Metrics

- Accuracy ranges from **70% to 85%** depending on the model and optimizer.
- Confusion matrices and classification reports are included in each notebook for detailed analysis.

---

## 🚀 Key Highlights

- Multiple CNN architectures compared for DR detection  
- Transfer learning applied on small dataset to improve results  
- Optimizer experimentation to find best training strategy  
- Modular and reproducible notebooks for easy experimentation  

---

## 🔗 How to Use

1. Clone the repository:  
```bash
git clone https://github.com/gagandeepsingh76/Diabetic-Eye-Retinopathy.git




