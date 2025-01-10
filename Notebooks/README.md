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

---

## **2. train_model_branching.ipynb**
- **Purpose**: Specializes the training by separating the prediction of each output into individual layers.
- **Key Features**:
  - **Separate Weights**:
    - Uses distinct output layers for each target variable (e.g., `LeftElbowYaw`, `LeftElbowRoll`, etc.).
    - This approach prevents weight sharing between predictions, allowing each output to learn independently.
  - Improves performance when different outputs have varying feature importance or behavior patterns.
  - Supports the addition of task-specific embeddings (e.g., grasp state or step size) for enhanced accuracy.

---

## **3. train_model_branching_pushing.ipynb**
- **Purpose**: Extends the branching approach to focus on **pushing tasks**, incorporating task-specific features and logic.
- **Key Features**:
  - Specializes predictions for pushing motions with additional features like:
    - Direction (`origin_x`, `origin_y`, `goal_x`, `goal_y`).
    - Task-specific embeddings and labels for pushing scenarios.
  - Uses advanced architectures:
    - Bidirectional LSTMs or attention mechanisms for capturing complex relationships in time-series data.
  - Evaluates task-specific metrics to fine-tune the model for real-world application.
