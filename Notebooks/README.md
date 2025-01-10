# **Notebooks Overview**

The **Notebooks** folder contains Jupyter notebooks for training and evaluating predictive models for robotic arm movements. These models focus on different scenarios and architectures to optimize predictions for specific tasks.

---

## **1. train_model.ipynb**
- **Purpose**: Provides a single, general-purpose model for predicting multiple robotic joint angles simultaneously.
- **Key Features**:
  - **Shared Weights**:
    - A single shared backbone processes all features.
    - The same set of layers is used to predict all outputs (e.g., `LeftElbowYaw`, `LeftElbowRoll`, `LeftShoulderRoll`, `LeftShoulderPitch`).
  - Simplifies the architecture for training on smaller or less complex datasets.
  - Visualizes loss curves for training and validation to ensure model convergence.
