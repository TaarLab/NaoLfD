import unittest
import requests
import json


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.base_url = "http://127.0.0.1:5000"
        self.image_path = "test.jpg"
        self.headers = {"Content-Type": "application/json"}

    def test_predict_objects(self):
        url = f"{self.base_url}/predict_objects"
        files = {'image': open(self.image_path, 'rb')}
        response = requests.post(url, files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('origin_x', data)
        self.assertIn('origin_y', data)
        self.assertIn('goal_x', data)
        self.assertIn('goal_y', data)
        print('predict_objects:', data)

    def test_predict_joints(self):
        url = f"{self.base_url}/predict_joints"
        payload = {
            'LeftElbowYaw': 0.5,
            'LeftElbowRoll': -0.3,
            'LeftShoulderRoll': 0.1,
            'LeftShoulderPitch': -0.5,
            'origin_x': 320,
            'origin_y': 240,
            'origin_z': 3.5,
            'goal_x': 330,
            'goal_y': 250,
            'goal_z': 3.6,
            'is_grasped': 1
        }
        response = requests.post(url, data=json.dumps(payload), headers=self.headers)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('LeftElbowYaw', data)
        self.assertIn('LeftElbowRoll', data)
        self.assertIn('LeftShoulderRoll', data)
        self.assertIn('LeftShoulderPitch', data)
        print('predict_joints:', data)


if __name__ == "__main__":
    unittest.main()
