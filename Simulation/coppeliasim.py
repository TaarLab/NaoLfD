import os

os.environ["KERAS_BACKEND"] = "torch"

import time
import math
import keras
import numpy as np
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

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
    'origin_x', 'origin_y', 'origin_z',
    'goal_x', 'goal_y', 'goal_z'
]

output_columns = [
    'LeftElbowYaw', 'LeftElbowRoll', 'LeftShoulderRoll', 'LeftShoulderPitch'
]

feature_shape = (10, len(feature_columns))  # window size, number of features


def update_robot_joints(sim, joint_handles, new_angles):
    for joint, angle in zip(joint_handles, new_angles):
        sim.setJointTargetPosition(joint, math.radians(angle))


def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def extract_features(sim):
    original_height_px = 144
    original_height_m = 0.54
    original_shoulder_distance_px = 200
    original_shoulder_distance_m = 0.41
    base_2d_origin = np.array([0.285, -0.02, -0.3])
    base_3d_origin = np.array([0.0, 0.2, -1.7])

    box = sim.getObject('/box')
    ball = sim.getObject('/ball')
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
    ball_pos = sim.getObjectPosition(ball, -1)
    box_pos = sim.getObjectPosition(box, -1)

    headers = [
        'LeftElbowYaw', 'LeftElbowRoll', 'RightElbowYaw', 'RightElbowRoll',
        'LeftShoulderRoll', 'LeftShoulderPitch', 'RightShoulderRoll', 'RightShoulderPitch',
        'LeftWristX', 'LeftWristY', 'LeftWristZ', 'RightWristX', 'RightWristY', 'RightWristZ',
        'LeftElbowX', 'LeftElbowY', 'LeftElbowZ', 'RightElbowX', 'RightElbowY', 'RightElbowZ',
        'LeftShoulderX', 'LeftShoulderY', 'LeftShoulderZ', 'RightShoulderX', 'RightShoulderY', 'RightShoulderZ',
        'origin_x', 'origin_y', 'origin_z', 'goal_x', 'goal_y', 'goal_z',
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

        640 - (base_2d_origin[0] + ball_pos[0]) * object_x_ratio,
        480 - (base_2d_origin[1] + ball_pos[2]) * object_y_ratio,
        (base_3d_origin[2] + ball_pos[1]) * z_ratio,
        640 - (base_2d_origin[0] + box_pos[0] - 0.00) * object_x_ratio,
        480 - (base_2d_origin[1] + box_pos[2] + 0.00) * object_y_ratio,
        (base_3d_origin[2] + box_pos[1]) * z_ratio,

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

    left_wrist = (data[32], data[33])
    box_position = (data[29], data[30])
    ball_position = (data[26], data[27])

    wrist_ball_distance = calculate_distance(left_wrist, ball_position)
    wrist_box_distance = calculate_distance(left_wrist, box_position)
    ball_box_distance = calculate_distance(ball_position, box_position)

    df = pd.DataFrame([dict(zip(headers, data))], columns=headers)

    df.at[0, 'WristBallDistance'] = wrist_ball_distance
    df.at[0, 'WristBoxDistance'] = wrist_box_distance
    df.at[0, 'BallBoxDistance'] = ball_box_distance

    return df


def prepare_input_features(features, feature_shape):
    num_rows, num_columns = features.shape
    num_rows_to_add = feature_shape[0] - num_rows
    new_rows_array = np.full((num_rows_to_add, num_columns), 0)
    prepared_features = np.concatenate((new_rows_array, features), axis=0)
    return np.expand_dims(prepared_features, axis=0)


def normalize_columns(df):
    column_ranges = {
        'LeftElbowYaw': (-120, 120), 'LeftElbowRoll': (-90, 0),
        'RightElbowYaw': (-120, 120), 'RightElbowRoll': (0, 90),
        'LeftShoulderRoll': (-45, 90), 'LeftShoulderPitch': (-120, 120),
        'RightShoulderRoll': (-90, 45), 'RightShoulderPitch': (-120, 120),
        'LeftWristX': (-1, 1), 'LeftWristY': (-1, 1), 'LeftWristZ': (0, 3),
        'RightWristX': (-1, 1), 'RightWristY': (-1, 1), 'RightWristZ': (0, 3),
        'LeftElbowX': (-1, 1), 'LeftElbowY': (-1, 1), 'LeftElbowZ': (0, 3),
        'RightElbowX': (-1, 1), 'RightElbowY': (-1, 1), 'RightElbowZ': (0, 3),
        'LeftShoulderX': (-1, 1), 'LeftShoulderY': (-1, 1), 'LeftShoulderZ': (0, 3),
        'RightShoulderX': (-1, 1), 'RightShoulderY': (-1, 1), 'RightShoulderZ': (0, 3),
        'origin_x': (0, 640), 'origin_y': (0, 480), 'origin_z': (0, 7),
        'goal_x': (0, 640), 'goal_y': (0, 480), 'goal_z': (0, 7),
        'LeftWristX_PX': (0, 640), 'LeftWristY_PX': (0, 480),
        'RightWristX_PX': (0, 640), 'RightWristY_PX': (0, 480),
        'LeftElbowX_PX': (0, 640), 'LeftElbowY_PX': (0, 480),
        'RightElbowX_PX': (0, 640), 'RightElbowY_PX': (0, 480),
        'LeftShoulderX_PX': (0, 640), 'LeftShoulderY_PX': (0, 480),
        'RightShoulderX_PX': (0, 640), 'RightShoulderY_PX': (0, 480),
        'WristBallDistance': (0, 800), 'WristBoxDistance': (0, 800), 'BallBoxDistance': (0, 800),
    }

    for column, (min_val, max_val) in column_ranges.items():
        df[column] = (df[column] - min_val) / (max_val - min_val)
        if min_val < 0:
            df[column] = 2 * df[column] - 1

    return df[feature_columns]


def restore_scale(predictions, label_ranges):
    restored_predictions = []
    for label in ['LeftElbowYaw', 'LeftElbowRoll', 'LeftShoulderRoll', 'LeftShoulderPitch']:
        min_val, max_val = label_ranges[label]
        normalized_value = predictions[label]

        if min_val < 0:
            original_value = ((normalized_value + 1) / 2) * (max_val - min_val) + min_val
        else:
            original_value = normalized_value * (max_val - min_val) + min_val

        restored_predictions.append(original_value)
    return np.array(restored_predictions)


def check_ball_in_box(sim, box_handle, ball_handle):
    box_min_x = sim.getObjectFloatParam(box_handle, sim.objfloatparam_objbbox_min_x)
    box_max_x = sim.getObjectFloatParam(box_handle, sim.objfloatparam_objbbox_max_x)

    box_pos = np.array(sim.getObjectPosition(box_handle, -1))[0]
    box_size = abs(box_max_x - box_min_x) + 0.02
    box_min = box_pos - box_size / 2
    box_max = box_pos + box_size / 2

    ball_pos = np.array(sim.getObjectPosition(ball_handle, -1))[0]

    return box_min <= ball_pos <= box_max


def run_simulation(trials=100):
    has_controlled = False
    client = RemoteAPIClient()
    sim = client.require('sim')

    success_count = 0

    for trial in range(trials):
        sim.startSimulation()

        joint_handles = [sim.getObject(f'/{joint_name}') for joint_name in [
            'LElbowYaw', 'LElbowRoll', 'LShoulderRoll', 'LShoulderPitch'
        ]]

        finger_joints = [sim.getObject(f'/NAO/{finger}') for finger in [
            'LThumbBase', 'LRFingerBase', 'LLFingerBase',
            'LThumbBase/joint', 'LRFingerBase/joint', 'LLFingerBase/joint',
            'LRFingerBase/joint/joint', 'LLFingerBase/joint/joint'
        ]]

        box_handle = sim.getObject('/box')
        ball_handle = sim.getObject('/ball')
        for finger_joint in finger_joints:
            sim.setJointTargetForce(finger_joint, -1)

        def prevent_collision_with_bounding_boxes_x(sim, box_handle, ball_handle):
            box_min_x = sim.getObjectFloatParam(box_handle, sim.objfloatparam_objbbox_min_x)
            box_max_x = sim.getObjectFloatParam(box_handle, sim.objfloatparam_objbbox_max_x)

            ball_min_x = sim.getObjectFloatParam(ball_handle, sim.objfloatparam_objbbox_min_x)
            ball_max_x = sim.getObjectFloatParam(ball_handle, sim.objfloatparam_objbbox_max_x)

            box_pos = np.array(sim.getObjectPosition(box_handle, -1))[0]
            box_size = abs(box_max_x - box_min_x)
            box_min = box_pos - box_size / 2
            box_max = box_pos + box_size / 2

            ball_pos = np.array(sim.getObjectPosition(ball_handle, -1))[0]
            ball_size = abs(ball_max_x - ball_min_x)
            ball_min = ball_pos - ball_size / 2
            ball_max = ball_pos + ball_size / 2

            return ball_max < box_min

        while True:
            current_box_position = sim.getObjectPosition(box_handle, -1)
            current_ball_position = sim.getObjectPosition(ball_handle, -1)

            box_x_adjustment = (0.31 - 0.24) * np.random.random() + 0.24
            new_box_position = [
                box_x_adjustment,
                current_box_position[1],
                current_box_position[2]
            ]

            ball_x_adjustment = (0.15 - 0.08) * np.random.random() + 0.08
            new_ball_position = [
                ball_x_adjustment,
                current_ball_position[1],
                current_ball_position[2]
            ]

            sim.setObjectPosition(box_handle, new_box_position)
            sim.setObjectPosition(ball_handle, new_ball_position)

            if prevent_collision_with_bounding_boxes_x(sim, box_handle, ball_handle):
                break

            sim.setObjectPosition(box_handle, current_box_position)
            sim.setObjectPosition(ball_handle, current_ball_position)

        old_restored_predictions = None
        counter = 0
        storage_list = []
        while True:
            features = extract_features(sim)
            normalized_features = normalize_columns(features)
            storage_list.append(normalized_features.values.astype(np.float64)[0])
            storage_list = storage_list[-feature_shape[0]:]
            prepared_features = prepare_input_features(np.array(storage_list), feature_shape)

            thumb_positions = [sim.getObjectPosition(finger_joint, -1) for finger_joint in finger_joints[:3]]
            avg_thumb_position = np.mean(thumb_positions, axis=0)

            ball_pos = sim.getObjectPosition(ball_handle, -1)
            palm_pos = sim.getObjectPosition(finger_joints[2], -1)
            wrist_pos = np.array(palm_pos)
            ball_pos = np.array(ball_pos)
            distance = np.linalg.norm(wrist_pos - ball_pos)

            if distance < 0.08:
                has_controlled = True
                sim.setObjectPosition(ball_handle, list(avg_thumb_position))
                for finger_joint in finger_joints:
                    sim.setJointTargetForce(finger_joint, 0.5)

            predictions = {}
            for joint, model in models["all"].items():
                prediction = model.predict([prepared_features, np.array([25]), np.array([1 if has_controlled else -1])])
                predictions[joint] = prediction[0][0]

            restored_predictions = restore_scale(predictions, label_ranges)

            if old_restored_predictions is None:
                old_restored_predictions = restored_predictions

            if abs(np.sum(restored_predictions - old_restored_predictions)) < 1 and has_controlled:
                counter += 1
            else:
                counter = 0

            if counter > 1 or distance > 2:
                for finger_joint in finger_joints:
                    sim.setJointTargetForce(finger_joint, -1)
                time.sleep(3)

                ball_pos = sim.getObjectPosition(ball_handle, -1)
                box_pos = sim.getObjectPosition(finger_joints[2], -1)
                ball_box_distance = np.linalg.norm(np.array(ball_pos) - np.array(box_pos))
                ball_in_box = check_ball_in_box(sim, box_handle, ball_handle)
                if ball_in_box and ball_box_distance < 0.5:
                    success_count += 1
                    print(f"Trial {trial + 1}/{trials}: Success!")
                else:
                    print(f"Trial {trial + 1}/{trials}: Failure.")
                break

            old_restored_predictions = restored_predictions
            update_robot_joints(sim, joint_handles, restored_predictions)
            time.sleep(0.1)

        sim.stopSimulation()
        time.sleep(1)

    success_rate = (success_count / trials) * 100
    print(f"Success rate: {success_rate:.2f}% over {trials} trials.")


if __name__ == "__main__":
    run_simulation()
