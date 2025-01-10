# Project Structure

The project consists of multiple modules, each serving a specific purpose in the overall system. Below is a breakdown of each project folder:

### 1. **Core**
   - **Purpose**: Core utilities and robotic control logic.
   - **Key Files**:
     - `Robot.cs`: Implements HTTP-based control of robotic arm movements.
     - `AngleCalculation.cs`: Contains algorithms for joint angle computation.

### 2. **Core.Test**
   - **Purpose**: Validates the accuracy of angle calculations and robotic movements.
   - **Key Files**:
     - `UnitTest.cs`: Contains tests for joint angle computations and robot movement routines.

### 3. **KinectSkeleton**
   - **Purpose**: Captures skeleton data from Kinect sensors and processes it for robotic motion mapping.
   - **Key Features**:
     - Tracks joint positions and writes processed data to CSV.
     - Generates visual overlays for joint movement.

### 4. **MediaPipeSkeleton**
   - **Purpose**: Uses Mediapipe for real-time human pose detection.
   - **Key Features**:
     - Extracts and processes pose landmarks for movement calculations.

### 5. **RealWorld**
   - **Purpose**: Integrates sensor data with robotic control in a real-world application.
   - **Key Features**:
     - Processes Kinect data to predict robot joint angles.
     - Sends movement commands to the robot via API calls.

---

## Dependencies

### Hardware
- **Kinect Sensor**: For capturing depth and color streams.
- **Robotic Arm**: Controlled through HTTP API.

---

## Installation

1. Restore dependencies:
   ```bash
   dotnet restore
   ```
2. Set up the Kinect sensor and ensure the required drivers are installed.
3. Configure the API endpoints in `Robot.cs` for robot control in `Core` Module.
