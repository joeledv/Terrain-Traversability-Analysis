# Terrain Traversability Analysis

## Installation

### Requirements

- Python: 3.10+ (recommended)
  - Notes: The listed dependencies (e.g., TensorFlow 2.11) were tested on Python 3.8–3.10. If you plan to use Python >= 3.11 or 3.13+, verify that all packages (especially TensorFlow, NumPy, and binary wheels) support that version before upgrading.
- pip
- Install dependencies from `requirements.txt`: `pip install -r requirements.txt`
    
## Pipeline

<p align="center">
<img width="1830" height="1016" alt="Captura de pantalla 2025-12-28 181331" src="https://github.com/user-attachments/assets/85244ca8-e3df-4502-8122-6442f2af7ce0" />
</p>

---

### System Overview

The development workflow follows a modular and reproducible pipeline composed of the following stages:

1. LiDAR data acquisition  
2. Preprocessing and filtering  
2.1. Geometric transformation (Bird’s Eye View)  
3. Feature extraction and vectorization  
4. Neural network training  
5. Semantic Segmentation 

Each stage was designed to be independent, scalable, and suitable for real-world deployment scenarios.

---

### 1. Data Acquisition

- Sensor: **Ouster OS1 3D LiDAR**
- Output: 3D point clouds (XYZ + radiometric attributes)
- Terrain classes:
  - Cobblestone
  - Asphalt
  - Concrete
  - Gravel
  - Grass

For each terrain type, multiple real-world locations were selected. Each site was recorded for approximately one minute, from which representative frames were extracted to build the dataset. All data were collected in semi-urban environments to ensure realistic variability.

---

### 2. Preprocessing

Before analysis, raw point clouds undergo a preprocessing stage that includes:

- Removal of points outside the region of interest
- Noise and outlier filtering
- Ground surface isolation

This step improves geometric consistency and reduces variability that could negatively impact model training.

---

#### 2.1. Geometric Transformation (Bird’s Eye View)

Due to the physical inclination of the LiDAR sensor during data acquisition, the raw point clouds are not aligned with the ground plane. To correct this:

- Homogeneous transformation matrices are applied
- Sensor tilt is compensated
- The ground plane is aligned with the X–Y plane
- A consistent Bird’s Eye View (BEV) representation is obtained

This transformation ensures spatial consistency across datasets captured at different locations and times.

#### Original Cloud Point

<p align="center">
<img width="503" height="459" alt="image" src="https://github.com/user-attachments/assets/caba0682-c9cd-4ef3-8664-6d83f794886c" />
</p>

#### Transformed Cloud Point

<p align="center">
<img width="622" height="546" alt="image" src="https://github.com/user-attachments/assets/8f97d29a-2cb7-49bb-8a68-d4c3f1fc30d9" />
</p>

---

### 3. Feature Extraction and Vectorization

From the transformed point clouds, local features are computed for each point to characterize terrain geometry and material properties. Feature extraction is based on a **Nearest Neighbors** approach to describe the local neighborhood of each point.

Two neighborhood sizes were evaluated:

- **N = 10**, capturing fine-grained local geometric variations  
- **N = 50**, capturing more global surface characteristics and increased spatial smoothing  

The extracted features include:

- Relative height
- Local height variance
- Surface roughness
- Point density
- Surface normal components
- LiDAR return intensity and reflectivity

Each point is represented as a feature vector, forming the input to the deep learning model. The comparison between N = 10 and N = 50 allowed evaluating the impact of neighborhood size on classification performance and robustness.

<p align="center">
<img width="3000" height="2964" alt="image" src="https://github.com/user-attachments/assets/90438798-ea89-4f45-a71a-0b0753045fa0" />
</p>

---

### 4. Classification Model

- Architecture: **Multilayer Perceptron (MLP)**
- Input: per-point feature vectors
- Output: terrain class label
- Training setup:
  - Loss function: categorical cross-entropy
  - Optimizer: Adam
  - Validation strategy: site-based split
  - Regularization: early stopping and normalization

The model performs **point-wise semantic segmentation** over the LiDAR point cloud.

---

### 5. Evaluation 

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

To assess generalization capability, the trained model was tested on terrain samples from locations not used during training. This validates robustness against changes in geometry, texture, and environmental conditions. Additionally, the results obtained using **N = 10 and N = 50** were compared to analyze the trade-off between local sensitivity and spatial stability.

#### N = 10
- Accuracy: 97.82%
- Precision: 97.78%
- Recall: 97.76%
- F1-score: 97.77%

#### N = 50
- Accuracy: 99.30%
- Precision: 99.28%
- Recall: 99.28%
- F1-score: 99.28%

--- 
### 6. Generalization

#### Cobblestone

<p align="center">
<img width="787" height="446" alt="image" src="https://github.com/user-attachments/assets/31ef4a7f-6143-4793-a2fd-adb7ead4eb44" />
</p>

<p align="center">
<img width="1507" height="644" alt="image" src="https://github.com/user-attachments/assets/0f838511-9893-4944-ac5b-141be37b7d48" />
</p>

#### Asphalt

<p align="center">
<img width="807" height="452" alt="image" src="https://github.com/user-attachments/assets/911e2ec3-fdc6-44d1-af94-a8a19dff5633" />
</p>

<p align="center">
<img width="1490" height="608" alt="image" src="https://github.com/user-attachments/assets/fd187ed7-9a60-4eee-a169-5a8f6f9c6765" />
</p>

### Concrete

<p align="center">
<img width="879" height="495" alt="image" src="https://github.com/user-attachments/assets/ae88f806-e767-413a-acf2-2a342ffd40ba" />
</p>

<p align="center">
<img width="1483" height="612" alt="image" src="https://github.com/user-attachments/assets/84531f5e-c844-4445-9383-703d3eec1e00" />
</p>

### Gravel

<p align="center">
<img width="913" height="513" alt="image" src="https://github.com/user-attachments/assets/cb1ba1ae-ea62-4079-a668-83111aaebf3e" />
</p>

<p align="center">
<img width="1487" height="599" alt="image" src="https://github.com/user-attachments/assets/acc42c48-5bea-4e10-9f0b-e8f7ad07ffdf" />
</p>

### Grass

<p align="center">
<img width="861" height="487" alt="image" src="https://github.com/user-attachments/assets/327d29f8-2cd8-4290-83d4-40db9c826d37" />
</p>

<p align="center">
<img width="1495" height="634" alt="image" src="https://github.com/user-attachments/assets/eecd1b89-5d5a-4ac6-8ff4-43e4419c9627" />
</p>

### 7. Ouster 3D LiDAR Integration on a Scale Jeep Rubicon

<div align="center">
  <img width="362" height="587" alt="image" src="https://github.com/user-attachments/assets/900348cc-e289-45e1-8f99-a51de45ab6c0" />
  <br><br>
  <img width="1047" height="591" alt="image" src="https://github.com/user-attachments/assets/a62dce97-8861-4f4c-bf36-655b8a4cbb64" />
</div>

---
### 7. Experiment Tracking & MLOps

To ensure reproducibility and efficiently manage model training, we integrated **MLflow** into our pipeline, centralizing all tracking workflows in the `MLOps/` directory.

Our model training and selection process involved the following steps:

* **Hyperparameter Optimization:** We utilized **Optuna** alongside MLflow to automatically tune the model's hyperparameters. Every trial was systematically logged, allowing us to easily compare different configurations through the MLflow UI dashboard.
* **Best Model Selection:** By analyzing the logged runs, we identified the best-performing model configuration, which achieved an outstanding **F1-Score of 0.9876**.
* **Final Retraining:** To maximize learning, the optimal hyperparameter configuration was used to retrain the model on the combined training and validation datasets.
* * **Evaluation & Interpretability:** We evaluated the final model using a confusion matrix to analyze class-wise performance. Additionally, we leveraged MLflow's tracking capabilities to extract **Feature Importance**, providing clear insights into which LiDAR geometric inputs had the greatest impact on the model's predictions.

<p align="center">
  <img width="45%" alt="Confusion Matrix" src="[https://github.com/user-attachments/assets/c7ae0930-740c-4d21-9cd4-4c85eaf7c616"]">
  <img width="45%" alt="Feature Importance" src="[https://github.com/user-attachments/assets/2513cbb8-0611-4b87-9527-71d9fd4ebd46]">
</p>
* **Model Export:** Finally, the best model was saved and exported in `.json` format, making it lightweight and ready for deployment.

**How to view the MLflow Dashboard:**
If you have the local tracking data (`mlruns/`), navigate to the project root in your terminal and run the following command to access the UI:

```bash
mlflow ui
