# MIT-BIH Arrhythmia Classification & Deep Learning Experiments

This repository contains a complete deep-learning pipeline developed for the **MIT-BIH Arrhythmia Dataset** and extended research experiments on **representation learning**, **neural collapse**, and **layer rotation (Layca)**.

The project includes:

- A full **end-to-end ECG arrhythmia classifier** using a Feedforward Neural Network.
- Exploration of **data imbalance**, **feature patterns**, and preprocessing strategies.
- Experiments with **Dropout**, **Batch Normalization**, and **Learning Rate Scheduling**.
- **UMAP visualisation** of learned representations across network layers.
- Reproduction and analysis of **Layca** (Layer Rotation) on CIFAR-10.
- A custom training system using **CutMix + Center Loss** to study feature geometry and robustness.

## Project Motivation

ECG arrhythmia classification is a widely studied problem, and the MIT-BIH dataset is a benchmark for heartbeat classification.

This project not only builds a strong arrhythmia classifier but also investigates how deep neural networks:

- Learn and separate features across layers  
- Respond to regularisation methods  
- Exhibit neural-collapse-like behaviour  
- Rotate weights during training (Layca)  
- Gain robustness under heavy augmentation (CutMix)

## Contents

### **Set 1: Forward Neural Network**
- Exploratory Data Analysis (EDA)
- Understanding class imbalance
- Feature distribution analysis
- Preprocessing: scaling, encoding, train–test split
- Weighted sampling for minority classes

### **Set 2: Model Improvements**
- Feedforward NN (128 → 64 → 32)
- CrossEntropy loss + Adam
- BatchNorm, Dropout, LR Scheduler
- TensorBoard logging  
- Early stopping + best model checkpoints

### **Set 3: Representation Visualisation**
- Registering forward hooks
- Extracting hidden-layer activations
- UMAP embeddings for fc1, fc2, fc3
- Analysis of representation evolution and separability

### **Set 4: High Distinction Tasks**
#### **4.1 Reproducing Layca**
- CIFAR-10 CNN baseline
- Custom Layca optimizer
- Layer rotation monitoring (cosine distance)
- Comparison with Carbonnelle & De Vleeschouwer (2019)
- Research gaps + theoretical connections

#### **4.2 Custom ML Solution**
A new training pipeline combining:

- **CutMix augmentation**  
- **Center Loss for compact features**  
- **ResNet-18 feature extractor**  
- **Dual-loss optimization**  
- **Rotation logging for early & final layers**

This method differs fundamentally from Layca by shaping feature geometry and label mixing rather than controlling optimizer rotation.


## Results Summary

### **Feedforward Neural Network**
- Achieved **~98% validation accuracy**
- Generalizes well due to:
  - Weighted sampling
  - BatchNorm + Dropout
  - LR scheduling
  - Early stopping

### **Representation Learning**
- fc1: highly entangled, low separability  
- fc2: partial clustering  
- fc3: strong class separation → aligns with high performance  
- Indicates hierarchical feature refinement

### **Layca Reproduction**
- Shallow CNN reached **~69% test accuracy**
- Lower than paper due to:
  - No full rotation achieved  
  - Simpler architecture  
  - Evidence rotation–generalization link persists  

### **CutMix + Center Loss**
- Stronger class compactness  
- Improved feature geometry  
- Better calibration  
- More stable angles of rotation compared to vanilla SGD


### Requirements

* Python 3.8+
* PyTorch
* NumPy, Pandas, Scikit-learn
* Matplotlib, Seaborn
* UMAP
* TorchVision
* TensorBoard


## Repository Structure

```
│── data/
│── models/
│── notebooks/
│── src/
│   ├── preprocess.py
│   ├── ffnn.py
│   ├── train_ffnn.py
│   ├── umap_visualize.py
│   ├── layca.py
│   ├── layca_train.py
│   ├── cutmix.py
│   ├── centerloss.py
│   ├── train_cutmix_centerloss.py
│── runs/ (TensorBoard logs)
│── best_model.pt
│── README.md
```

## References

MIT-BIH Arrhythmia Database — PhysioNet
CIFAR-10 Dataset — Krizhevsky et al.

