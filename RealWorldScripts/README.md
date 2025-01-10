# **RealWorldScripts Overview**

The **RealWorldScripts** folder contains scripts designed to facilitate the integration of trained models and sensor data into real-world robotic systems. These scripts focus on robotic arm control, real-time data processing, and predictive modeling to achieve tasks such as object detection, pose estimation, and robotic joint movement prediction.

---

## **Scripts**

### **1. predictor_server.py**
- **Purpose**: Serves as the main backend for handling predictions and controlling the robotic arm.
- **Key Features**:
  - Hosts endpoints for:
    - Object detection (`/predict_objects`): Identifies objects in captured frames and provides their positions (`origin_x`, `origin_y`, etc.).
    - Joint predictions (`/predict_joints`): Computes robotic joint angles based on the provided feature data.
  - Integrates pre-trained models for joint prediction.
  - Uses YOLOWorld for object detection.

---

### **2. nao_server.py**
- **Purpose**: Manages the robotic arm's movements and communicates with the NAO robot's motion API.
- **Key Features**:
  - Provides endpoints for:
    - Moving the robot arm to specified angles.
    - Retrieving the current joint angles.
  - Implements safety mechanisms to clamp joint movements within predefined limits.

---

### **3. predictor_server_test.py**
- **Purpose**: Contains unit tests for validating the functionality of `predictor_server.py`.
- **Key Features**:
  - Tests endpoints for:
    - Object detection (`/predict_objects`).
    - Joint predictions (`/predict_joints`).
  - Ensures the server responds correctly with valid outputs for various scenarios.
