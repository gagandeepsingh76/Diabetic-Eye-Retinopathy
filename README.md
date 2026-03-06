# 🧠 Diabetic Eye Retinopathy Detection

This repository contains multiple deep learning experiments for **Diabetic Eye Retinopathy (DR) classification** using state-of-the-art Convolutional Neural Network (CNN) architectures. The goal is to develop automated systems to detect DR from retinal images with high accuracy, exploring various models, optimizers, and training strategies.

---

# 📌 Overview

Diabetic Retinopathy is a diabetes-related complication that affects the eyes and can lead to blindness if untreated. Early detection is critical, and automated DR classification can assist ophthalmologists by providing fast and reliable screening.

This project implements and compares several CNN models on a **dataset of 1,500 retinal images** divided into **2 classes** (DR and No DR). Model accuracies range from **70% to 85%**, depending on architecture and training strategies.

---

# 📂 Project Structure

```
Diabetic-Eye-Retinopathy
│
├── enhance_brightness
│   ├── b0_2.ipynb
│   ├── b0_4.ipynb
│   ├── b0_6.ipynb
│   ├── b0_8.ipynb
│   ├── b1.ipynb
│   ├── b1_2.ipynb
│   ├── b1_4.ipynb
│   ├── b1_6.ipynb
│   ├── b1_8.ipynb
│   ├── b2.ipynb
│   ├── b2_2.ipynb
│   ├── b2_4.ipynb
│   ├── b2_6.ipynb
│   ├── b2_8.ipynb
│   └── b3.ipynb
│
├── MobileNetV3.ipynb
├── efficientnetB0.ipynb
├── inceptionv3.ipynb
├── inceptionv3__.ipynb
├── inceptionv3_enh.ipynb
├── resnet152ADAM.ipynb
├── resnet152ADAM1.ipynb
├── README.md
```

---

# 🧩 Dataset

- **Size:** 1,500 retinal images  
- **Classes:** 2 (DR, No DR)

### Preprocessing Steps

- Resizing images to match model input shapes
- Normalization
- Data augmentation:
  - Rotation
  - Flipping
  - Zoom

---

# 📌 Implemented Models

| Model | Key Features | Accuracy Range |
|------|--------------|---------------|
| InceptionV3 | Baseline CNN | 72–78% |
| InceptionV3 Variation | Tweaked layers and training | 73–79% |
| Enhanced InceptionV3 | Advanced training strategies and fine-tuning | 78–83% |
| ResNet152 + Adam | Deep residual network | 75–82% |
| ResNet152 + Modified Adam | Optimized Adam variants | 76–83% |
| ResNet152 + Adamax | Adamax optimizer | 74–81% |
| ResNet152 + Nadam | Nadam optimizer | 75–84% |
| ResNet152 + RMSprop | RMSprop optimizer | 73–80% |
| EfficientNet-B0 | Lightweight and efficient architecture | 70–78% |
| MobileNetV3 | Mobile-friendly and fast inference | 70–77% |

---

# 🛠️ Tech Stack

**Programming Language**

- Python 🐍

**Deep Learning Framework**

- TensorFlow
- Keras

**Data Processing**

- NumPy
- Pandas

**Visualization**

- Matplotlib
- Seaborn

**Model Evaluation**

- Scikit-learn

---

# 📊 Performance Metrics

- Accuracy ranges from **70% to 85%** depending on the model and optimizer.
- Confusion matrices and classification reports are included in each notebook for detailed analysis.

---

# 📈 Sample Results

<p align="center">
  <img width="567" height="490" alt="image" src="https://github.com/user-attachments/assets/273f3958-9c07-4aed-afbb-efdaf9132db9" />
</p>

---

# 🚀 Key Highlights

- Multiple CNN architectures compared for DR detection  
- Transfer learning applied on a small dataset to improve results  
- Optimizer experimentation to find the best training strategy  
- Modular and reproducible notebooks for easy experimentation  

---

# 🔗 How to Use

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/gagandeepsingh76/Diabetic-Eye-Retinopathy.git
```

### 2️⃣ Navigate to the Project Folder

```bash
cd Diabetic-Eye-Retinopathy
```

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Notebooks

Open the notebooks using **Jupyter Notebook or Google Colab** and run the cells to train or evaluate the models.

---

# 👨‍💻 Author

**Gagandeep Singh**

Computer Science Student  
Interested in **Artificial Intelligence, Computer Vision, and Deep Learning**

GitHub:  
https://github.com/gagandeepsingh76

---

# ⭐ Support

If you find this project useful, please consider giving it a **star ⭐ on GitHub**.
