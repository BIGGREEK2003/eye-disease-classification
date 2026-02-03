# Eye Disease Classification using Deep Learning

This project implements a multi-class image classification system to detect common eye diseases from medical imaging. It leverages various state-of-the-art Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to provide accurate diagnostic insights.

## Dataset
The project uses an eye disease dataset categorized into four classes:
- **Cataract**
- **Diabetic Retinopathy**
- **Glaucoma**
- **Normal**

The dataset is automatically split into training (75%), validation (12.5%), and test (12.5%) sets.

## Project Structure
- Data splitting and organization.
- GPU-accelerated data augmentation (flipping, rotation, zoom, etc.).
- Custom data pipeline using `tf.data` for high performance.
- Implementation of multiple deep learning architectures for comparison.
- Two-stage training process: Head warm-up followed by full model fine-tuning.
- Comprehensive evaluation using ROC curves, AUC, and classification reports.

## Models Evaluated
- **EfficientNetB3**
- **ResNet50**
- **VGG19**
- **MobileNetV3Large**
- **InceptionV3**
- **Vision Transformer (ViT-B16)**
- **Vision Transformer (ViT-B32)**

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- `vit-keras` for Vision Transformer support
- ## Dataset
The dataset used in this project is the **Eye Diseases Classification** dataset from Kaggle. It contains retinal fundus images for four categories: Cataract, Diabetic Retinopathy, Glaucoma, and Normal.

* **Source:** [Kaggle - Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
* **Format:** Image files (.jpg) organized into folders by class.
* **Total Images:** ~4,217 images.

### Installation
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn vit-keras
