import os
import keras
import numpy as np
import pandas as pd
from ultralytics import YOLOWorld
from flask import Flask, request, jsonify

os.environ["KERAS_BACKEND"] = "torch"

model = YOLOWorld('yolov8s-worldv2.pt')

app = Flask(__name__)

config = {
    'red ball': ['red ball', 'apple', 'red apple'],
    'green box': ['green box', 'green lego', 'green object']
}

# Load the trained models
models = {
    'LeftElbowYaw': keras.models.load_model('../model/best_model_LeftElbowYaw_all.keras'),
    'LeftElbowRoll': keras.models.load_model('../model/best_model_LeftElbowRoll_all.keras'),
    'LeftShoulderRoll': keras.models.load_model('../model/best_model_LeftShoulderRoll_all.keras'),
    'LeftShoulderPitch': keras.models.load_model('../model/best_model_LeftShoulderPitch_all.keras')
}

label_ranges = {
    'LeftElbowYaw': (-120, 120), 'LeftElbowRoll': (-90, 0),
    'LeftShoulderRoll': (-45, 90), 'LeftShoulderPitch': (-120, 120),
}

feature_columns = [
    'LeftElbowYaw', 'LeftElbowRoll', 'LeftShoulderRoll', 'LeftShoulderPitch',
    'origin_x', 'origin_y', 'origin_z',
    'goal_x', 'goal_y', 'goal_z',
    'is_grasped'
]

output_columns = [
    'LeftElbowYaw', 'LeftElbowRoll', 'LeftShoulderRoll', 'LeftShoulderPitch'
]

feature_shape = (10, len(feature_columns))  # window size, number of features


def normalize_columns(df):
    column_ranges = {
        'LeftElbowYaw': (-120, 120), 'LeftElbowRoll': (-90, 0),
        'LeftShoulderRoll': (-45, 90), 'LeftShoulderPitch': (-120, 120),
        'origin_x': (0, 640), 'origin_y': (0, 480), 'origin_z': (0, 7),
        'goal_x': (0, 640), 'goal_y': (0, 480), 'goal_z': (0, 7),
        'is_grasped': (-1, 1)
    }

    for column, (min_val, max_val) in column_ranges.items():
        df[column] = (df[column] - min_val) / (max_val - min_val)
        if min_val < 0:
            df[column] = 2 * df[column] - 1

    df = df[feature_columns]

    return df


def restore_scale(predictions, label_ranges):
    restored_predictions = []
    for label in ['LeftElbowYaw', 'LeftElbowRoll', 'LeftShoulderRoll', 'LeftShoulderPitch']:
        min_val, max_val = label_ranges[label]
        normalized_value = predictions[label]    # Indexing across time for each feature

        if min_val < 0:
            original_value = ((normalized_value + 1) / 2) * (max_val - min_val) + min_val
        else:
            original_value = normalized_value * (max_val - min_val) + min_val

        restored_predictions.append(original_value)
    return np.array(restored_predictions)  # Convert list to array for vectorized operations


def prepare_input_features(features, feature_shape):
    """ Prepare feature matrix for the model, filling unknown timesteps with -1 """
    num_rows, num_columns = features.shape
    num_rows_to_add = feature_shape[0] - num_rows
    new_rows_array = np.full((num_rows_to_add, num_columns), 0)
    prepared_features = np.concatenate((new_rows_array, features), axis=0)
    return np.expand_dims(prepared_features, axis=0)  # Add batch dimension for Keras


def detect_objects(image_path, model, config, confidence=0.23, main=False):
    detected_objects = {}

    primary_terms = [alternatives[0] for alternatives in config.values()]
    model.set_classes(primary_terms)
    results = model.predict(image_path, conf=confidence)

    if len(results[0].boxes) <= len(primary_terms):
        for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            object_name = list(config.keys())[int(cls_id)]
            x_min, y_min, x_max, y_max = box[:4]
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            detected_objects[object_name] = (center_x.item(), center_y.item())

    if not main:
        return detected_objects

    for object_name, prompts in config.items():
        while object_name not in detected_objects:
            if confidence >= 0.12:
                confidence -= 0.1
                detected_objects = detect_objects(image_path, model, {object_name: prompts},
                                                  confidence) | detected_objects
            elif len(prompts) > 1:
                confidence = 0.23
                prompts = prompts[1:]
                while object_name not in detected_objects:
                    if confidence >= 0.01:
                        detected_objects = detect_objects(image_path, model, {object_name: prompts},
                                                          confidence) | detected_objects
                        confidence -= 0.1
                    else:
                        break
            else:
                break

    return detected_objects

goal = None

@app.route('/predict_objects', methods=['POST'])
def predict_objects():
    global goal
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_path = os.path.join('temp', image.filename)
    image.save(image_path)

    detected_objects = detect_objects(image_path, model, config, 0.23, True)

    if detected_objects.get('green box', [None, None])[0] is not None:
        del config['green box']
        goal = [detected_objects.get('green box', [None, None])[0], detected_objects.get('green box', [None, None])[1]]

    response = {
        'origin_x': detected_objects.get('red ball', [None, None])[0],
        'origin_y': detected_objects.get('red ball', [None, None])[1],
        'goal_x': detected_objects.get('green box', [None, None])[0] if goal is None else goal[0],
        'goal_y': detected_objects.get('green box', [None, None])[1] if goal is None else goal[1]
    }

    os.remove(image_path)  # Clean up the saved image
    return jsonify(response)


storage_list = []
@app.route('/predict_joints', methods=['POST'])
def predict_joints():
    global storage_list
    data = request.json

    if not data or not all(col in data for col in feature_columns):
        return jsonify({'error': 'Invalid input'}), 400

    features = pd.DataFrame([data])
    normalized_features = normalize_columns(features)
    storage_list.append(normalized_features.values.astype(np.float64)[0])
    storage_list = storage_list[-feature_shape[0]:]
    prepared_features = prepare_input_features(np.array(storage_list), feature_shape)

    predictions = {}
    for joint, model in models.items():
        prediction = model.predict(prepared_features)
        predictions[joint] = prediction[0][0]

    restored_predictions = restore_scale(predictions, label_ranges)

    response = {col: val for col, val in zip(output_columns, restored_predictions)}

    return jsonify(response)


if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(debug=True)
