# 🧠 Alzheimer's Disease Detection Using MRI and Deep Learning

**Authors:**  
- Ruthik Raj Nataraja – nataraja.r@northeastern.edu  


---

## 🔍 Project Overview

This project aims to detect **Alzheimer’s Disease** using deep learning techniques on MRI scans. We employed **EfficientNetB0**, a state-of-the-art image classification model, for feature extraction and classification. The dataset includes four classes of dementia:
- Non-Demented (Class 0)  
- Very Mild Demented (Class 1)  
- Mild Demented (Class 2)  
- Moderate Demented (Class 3)

Our deep learning pipeline enables early and accurate diagnosis of Alzheimer's disease, facilitating timely medical intervention and better patient outcomes.

---

## 📁 Dataset

- **Source:** [Hugging Face - Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI)  
- **Size:** 6,400 images  
- **Split:**  
  - Training set: 5,120 images  
  - Test set: 1,280 images  
- **Classes:**  
  - Non-Demented  
  - Very Mild Demented  
  - Mild Demented  
  - Moderate Demented

---

## ⚙️ Methodology

### ✅ Preprocessing
- Image resizing
- Normalization
- Image augmentation (rotation, flipping, shearing, zooming)
- Stratified train-test split
- Batch normalization, dropout, L1 & L2 regularization to prevent overfitting

### 🧠 Models Compared
1. **CNN without augmentation** – Accuracy: 49.8%  
2. **CNN with image augmentation** – Accuracy: 54.38%  
3. **EfficientNetB0 (pre-trained on ImageNet)** – Accuracy: **94.5%**

### 📈 Best Performing Model: EfficientNetB0
- Transfer learning via ImageNet
- Dropout, batch normalization, regularization
- Adamax optimizer
- Achieved **94.5%** accuracy on validation set

---

## 📊 Results Summary

| Model                        | Accuracy (%) |
|-----------------------------|--------------|
| CNN (no augmentation)       | 49.8%        |
| CNN (with augmentation)     | 54.38%       |
| EfficientNetB0              | **94.5%**    |

EfficientNetB0 significantly outperformed traditional CNN models, demonstrating the importance of transfer learning and proper regularization.

---

## 🚧 Limitations

- **Limited Dataset Diversity**: May impact generalizability.
- **Black-Box Model**: Low interpretability of decision-making.
- **Real-World Applicability**: Performance may vary in clinical settings.
- **High Computational Requirements**: Limits deployment in resource-constrained environments.

---

## 🔬 Future Work

- Integrate additional modalities (e.g., PET scans, genetic data)
- Explore advanced models (e.g., Vision Transformers, GNNs)
- Real-time classification and deployment in clinical pipelines
- Improve model interpretability (e.g., Grad-CAM)

---

## 📚 References

1. Odusami, M. et al. (2023). *Pixel-Level Fusion Approach with Vision Transformer for Early Detection of Alzheimer’s Disease*. Electronics.
2. Sharma, S. et al. (2022). *CNN with VGG16 Feature Extractor for Alzheimer Detection*. Measurement Sensors.
3. Khan, R. et al. (2022). *Transfer Learning for Multiclass Classification of Alzheimer’s Disease*. Frontiers in Neuroscience.
4. Kavitha, C. et al. (2022). *Early-Stage Alzheimer’s Prediction using ML*. Frontiers in Public Health.
5. Chen, H. (2022). *Alzheimer’s Detection using CNN and Random Forest*. Highlights in Science, Engineering and Technology.

---

## ✅ Conclusion

This project successfully demonstrates the application of **deep learning and transfer learning** in classifying Alzheimer’s disease stages from MRI images. **EfficientNetB0** achieved outstanding results, showcasing its potential for use in medical diagnostics. Our findings highlight the need for continued exploration in this domain to improve accuracy, interpretability, and real-world deployment.

---
