# Terrain-Traversability-Analysis

## Development

This section describes the complete development pipeline implemented to transform raw 3D LiDAR data into structured information for automatic terrain classification using deep learning.

---

### System Overview

The development workflow follows a modular and reproducible pipeline composed of the following stages:

1. LiDAR data acquisition  
2. Preprocessing and filtering  
3. Geometric transformation (Bird’s Eye View)  
4. Feature extraction and vectorization  
5. Neural network training  
6. Evaluation and generalization  

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

### 3. Geometric Transformation (Bird’s Eye View)

Due to the physical inclination of the LiDAR sensor during data acquisition, the raw point clouds are not aligned with the ground plane. To correct this:

- Homogeneous transformation matrices are applied
- Sensor tilt is compensated
- The ground plane is aligned with the X–Y plane
- A consistent Bird’s Eye View (BEV) representation is obtained

This transformation ensures spatial consistency across datasets captured at different locations and times.

---

### 4. Feature Extraction and Vectorization

From the transformed point clouds, local features are computed for each point to characterize terrain geometry and material properties. Feature extraction is based on a **K-Nearest Neighbors (KNN)** approach to describe the local neighborhood of each point.

Two neighborhood sizes were evaluated:

- **K = 10**, capturing fine-grained local geometric variations  
- **K = 50**, capturing more global surface characteristics and increased spatial smoothing  

The extracted features include:

- Relative height
- Local height variance
- Surface roughness
- Point density
- Surface normal components
- LiDAR return intensity and reflectivity

Each point is represented as a feature vector, forming the input to the deep learning model. The comparison between K = 10 and K = 50 allowed evaluating the impact of neighborhood size on classification performance and robustness.

---

### 5. Classification Model

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

### 6. Evaluation and Generalization

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

To assess generalization capability, the trained model was tested on terrain samples from locations not used during training. This validates robustness against changes in geometry, texture, and environmental conditions. Additionally, the results obtained using **K = 10 and K = 50** were compared to analyze the trade-off between local sensitivity and spatial stability.

---

### Repository Structure (Development)



