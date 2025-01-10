import os

os.environ["KERAS_BACKEND"] = "torch"

import time
import math
import keras
import numpy as np
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from Simulation.coppeliasim import update_robot_joints, normalize_columns, prepare_input_features, \
    restore_scale

models = {
    'all': {
        'LeftElbowYaw': keras.models.load_model('../model/best_model_LeftElbowYaw_all.keras'),
        'LeftElbowRoll': keras.models.load_model('../model/best_model_LeftElbowRoll_all.keras'),
        'LeftShoulderRoll': keras.models.load_model('../model/best_model_LeftShoulderRoll_all.keras'),
        'LeftShoulderPitch': keras.models.load_model('../model/best_model_LeftShoulderPitch_all.keras')
    }
}

label_ranges = {
    'LeftElbowYaw': (-120, 120), 'LeftElbowRoll': (-90, 0),
    'LeftShoulderRoll': (-45, 90), 'LeftShoulderPitch': (-120, 120),
}
feature_columns = [
    'LeftElbowYaw', 'LeftElbowRoll', 'LeftShoulderRoll', 'LeftShoulderPitch',
    'origin_x', 'origin_y', 'origin_z'
]
output_columns = [
    'LeftElbowYaw', 'LeftElbowRoll', 'LeftShoulderRoll', 'LeftShoulderPitch'
]

feature_shape = (10, len(feature_columns))  # window size, number of features

def extract_features(sim):
    original_height_px = 144
    original_height_m = 0.54
    original_shoulder_distance_px = 200
    original_shoulder_distance_m = 0.41
    base_2d_origin = np.array([0.285, -0.02, -0.3])
    base_3d_origin = np.array([0.0, 0.2, -1.7])

    cuboid = sim.getObject('/Cuboid')
    r_elbow_yaw = sim.getObject('/RElbowYaw')
    l_elbow_yaw = sim.getObject('/LElbowYaw')
    r_wrist_yaw = sim.getObject('/RWristYaw')
    l_wrist_yaw = sim.getObject('/LWristYaw')
    r_elbow_roll = sim.getObject('/RElbowRoll')
    l_elbow_roll = sim.getObject('/LElbowRoll')
    r_shoulder_roll = sim.getObject('/RShoulderRoll')
    l_shoulder_roll = sim.getObject('/LShoulderRoll')
    r_shoulder_pitch = sim.getObject('/RShoulderPitch')
    l_shoulder_pitch = sim.getObject('/LShoulderPitch')
    r_hip_roll_link = sim.getObject('/RHipYawPitch/hip_roll_link_respondable')
    l_hip_roll_link = sim.getObject('/LHipYawPitch/hip_roll_link_respondable')

    r_hip_roll_link_pos = sim.getObjectPosition(r_hip_roll_link, -1)
    l_hip_roll_link_pos = sim.getObjectPosition(l_hip_roll_link, -1)

    foot_pos = [(r_hip_roll_link_pos[i] + l_hip_roll_link_pos[i]) / 2 for i in range(3)]
    head_pos = sim.getObjectPosition(sim.getObject('/HeadYaw'), -1)
    robot_height = (np.array(head_pos) - np.array(foot_pos))[2]

    robot_shoulder_distance = sim.getObjectPosition(l_shoulder_pitch, r_shoulder_pitch)[2]

    object_x_ratio = original_shoulder_distance_px / robot_shoulder_distance
    object_y_ratio = original_height_px / robot_height

    x_ratio = original_shoulder_distance_m / robot_shoulder_distance
    y_ratio = original_height_m / robot_height
    z_ratio = -1

    r_shoulder_pos = sim.getObjectPosition(r_shoulder_pitch, -1)
    l_shoulder_pos = sim.getObjectPosition(l_shoulder_pitch, -1)
    r_elbow_pos = sim.getObjectPosition(r_elbow_yaw, -1)
    l_elbow_pos = sim.getObjectPosition(l_elbow_yaw, -1)
    r_wrist_pos = sim.getObjectPosition(r_wrist_yaw, -1)
    l_wrist_pos = sim.getObjectPosition(l_wrist_yaw, -1)
    cuboid_pos = sim.getObjectPosition(cuboid, -1)

    headers = [
        'LeftElbowYaw', 'LeftElbowRoll', 'RightElbowYaw', 'RightElbowRoll',
        'LeftShoulderRoll', 'LeftShoulderPitch', 'RightShoulderRoll', 'RightShoulderPitch',
        'LeftWristX', 'LeftWristY', 'LeftWristZ', 'RightWristX', 'RightWristY', 'RightWristZ',
        'LeftElbowX', 'LeftElbowY', 'LeftElbowZ', 'RightElbowX', 'RightElbowY', 'RightElbowZ',
        'LeftShoulderX', 'LeftShoulderY', 'LeftShoulderZ', 'RightShoulderX', 'RightShoulderY', 'RightShoulderZ',
        'origin_x', 'origin_y', 'origin_z',
        'LeftWristX_PX', 'LeftWristY_PX',
        'RightWristX_PX', 'RightWristY_PX',
        'LeftElbowX_PX', 'LeftElbowY_PX',
        'RightElbowX_PX', 'RightElbowY_PX',
        'LeftShoulderX_PX', 'LeftShoulderY_PX',
        'RightShoulderX_PX', 'RightShoulderY_PX'
    ]
    data = [
        int(math.degrees(sim.getJointPosition(l_elbow_yaw))),
        int(math.degrees(sim.getJointPosition(l_elbow_roll))),
        int(math.degrees(sim.getJointPosition(r_elbow_yaw))),
        int(math.degrees(sim.getJointPosition(r_elbow_roll))),
        int(math.degrees(sim.getJointPosition(l_shoulder_roll))),
        int(math.degrees(sim.getJointPosition(l_shoulder_pitch))),
        int(math.degrees(sim.getJointPosition(r_shoulder_roll))),
        int(math.degrees(sim.getJointPosition(r_shoulder_pitch))),
        # Wrist positions
        (base_3d_origin[0] - l_wrist_pos[0]) * x_ratio,
        (base_3d_origin[1] - l_wrist_pos[2]) * y_ratio,
        (base_3d_origin[2] - l_wrist_pos[1]) * z_ratio,
        (base_3d_origin[0] - r_wrist_pos[0]) * x_ratio,
        (base_3d_origin[1] - r_wrist_pos[2]) * y_ratio,
        (base_3d_origin[2] - r_wrist_pos[1]) * z_ratio,
        # Elbow positions
        (base_3d_origin[0] - l_elbow_pos[0]) * x_ratio,
        (base_3d_origin[1] - l_elbow_pos[2]) * y_ratio,
        (base_3d_origin[2] - l_elbow_pos[1]) * z_ratio,
        (base_3d_origin[0] - r_elbow_pos[0]) * x_ratio,
        (base_3d_origin[1] - r_elbow_pos[2]) * y_ratio,
        (base_3d_origin[2] - r_elbow_pos[1]) * z_ratio,
        # Shoulder positions
        (base_3d_origin[0] - l_shoulder_pos[0]) * x_ratio,
        (base_3d_origin[1] - l_shoulder_pos[2]) * y_ratio,
        (base_3d_origin[2] - l_shoulder_pos[1]) * z_ratio,
        (base_3d_origin[0] - r_shoulder_pos[0]) * x_ratio,
        (base_3d_origin[1] - r_shoulder_pos[2]) * y_ratio,
        (base_3d_origin[2] - r_shoulder_pos[1]) * z_ratio,

        640 - (base_2d_origin[0] + cuboid_pos[0]) * object_x_ratio,
        480 - (base_2d_origin[1] + cuboid_pos[2]) * object_y_ratio,
        (base_3d_origin[2] + cuboid_pos[1]) * z_ratio,

        # Wrist positions
        (base_2d_origin[0] - l_wrist_pos[0]) * object_x_ratio,
        (base_2d_origin[1] - l_wrist_pos[2]) * object_y_ratio,
        (base_2d_origin[0] - r_wrist_pos[0]) * object_x_ratio,
        (base_2d_origin[1] - r_wrist_pos[2]) * object_y_ratio,
        # Elbow positions
        (base_2d_origin[0] - l_elbow_pos[0]) * object_x_ratio,
        (base_2d_origin[1] - l_elbow_pos[2]) * object_y_ratio,
        (base_2d_origin[0] - r_elbow_pos[0]) * object_x_ratio,
        (base_2d_origin[1] - r_elbow_pos[2]) * object_y_ratio,
        # Shoulder positions
        (base_2d_origin[0] - l_shoulder_pos[0]) * object_x_ratio,
        (base_2d_origin[1] - l_shoulder_pos[2]) * object_y_ratio,
        (base_2d_origin[0] - r_shoulder_pos[0]) * object_x_ratio,
        (base_2d_origin[1] - r_shoulder_pos[2]) * object_y_ratio
    ]

    df = pd.DataFrame([dict(zip(headers, data))], columns=headers)

    return df

def run_simulation():
    has_controlled = False
    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.startSimulation()

    joint_handles = [sim.getObject(f'/{joint_name}') for joint_name in [
        'LElbowYaw', 'LElbowRoll', 'LShoulderRoll', 'LShoulderPitch'
    ]]

    finger_joints = [sim.getObject(f'/NAO/{finger}') for finger in [
        'LThumbBase', 'LRFingerBase', 'LLFingerBase',
        'LThumbBase/joint', 'LRFingerBase/joint', 'LLFingerBase/joint',
        'LRFingerBase/joint/joint', 'LLFingerBase/joint/joint'
    ]]
    for finger_joint in finger_joints:
        sim.setJointTargetForce(finger_joint, 0.5)

    storage_list = []
    while True:
        features = extract_features(sim)

        thumb_positions = [sim.getObjectPosition(finger_joint, -1) for finger_joint in finger_joints[:3]]
        avg_thumb_position = np.mean(thumb_positions, axis=0)

        cuboid_handle = sim.getObject('/Cuboid')
        cuboid_pos = sim.getObjectPosition(cuboid_handle, -1)
        palm_pos = sim.getObjectPosition(finger_joints[2], -1)
        wrist_pos = np.array(palm_pos)
        cuboid_pos = np.array(cuboid_pos)
        distance = np.linalg.norm(wrist_pos - cuboid_pos)

        if distance < 0.08:
            has_controlled = True
            sim.setObjectPosition(cuboid_handle, list(avg_thumb_position))
            cuboid_pos[0] -= 0.01
            sim.setObjectPosition(cuboid_handle, list(cuboid_pos))

        normalized_features = normalize_columns(features)
        storage_list.append(normalized_features.values.astype(np.float64)[0])
        storage_list = storage_list[-feature_shape[0]:]
        prepared_features = prepare_input_features(np.array(storage_list), feature_shape)
        predictions = {}

        n = 0
        for joint, model in models["all"].items():
            prediction = model.predict([prepared_features, np.array([25]), np.array([1 if has_controlled else -1])])
            predictions[joint] = prediction[0][n]

        restored_predictions = restore_scale(predictions, label_ranges)
        new_angles = restored_predictions
        update_robot_joints(sim, joint_handles, new_angles)
        time.sleep(0.5)


if __name__ == "__main__":
    run_simulation()