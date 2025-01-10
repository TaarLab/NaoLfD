import math
import socket
import almath
from naoqi import ALProxy
import re
import json

ip = "192.168.101.146"
motionProxy = ALProxy("ALMotion", ip, 9559)

TO_RAD = almath.TO_RAD
TO_DEG = almath.TO_DEG

joint_limits = {
    'LElbowYaw': {'min': -119.5, 'max': 119.5},
    'LElbowRoll': {'min': -88.5, 'max': -2},
    'LShoulderRoll': {'min': -18, 'max': 76},
    'LShoulderPitch': {'min': -119.5, 'max': 119.5},
    'RElbowYaw': {'min': -119.5, 'max': 119.5},
    'RElbowRoll': {'min': 2, 'max': 88.5},
    'RShoulderRoll': {'min': -76, 'max': 18},
    'RShoulderPitch': {'min': -119.5, 'max': 119.5},
    'LWristYaw': {'min': -104.5, 'max': 104.5},
    'RWristYaw': {'min': -104.5, 'max': 104.5}
}

def clamp_angle(joint_name, angle):
    joint_limit = joint_limits[joint_name]
    return max(joint_limit['min'], min(joint_limit['max'], angle))

def move_arms(left_shoulder_pitch, left_shoulder_roll, left_elbow_yaw, left_elbow_roll,
              right_shoulder_pitch, right_shoulder_roll, right_elbow_yaw, right_elbow_roll):
    motionProxy.setStiffnesses("Body", 1.0)
    names = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "RShoulderPitch", "RShoulderRoll",
             "RElbowYaw", "RElbowRoll"]
    speed = 0.05

    # Clamp angles within their joint limits
    left_shoulder_pitch = clamp_angle('LShoulderPitch', left_shoulder_pitch)
    left_shoulder_roll = clamp_angle('LShoulderRoll', left_shoulder_roll)
    left_elbow_yaw = clamp_angle('LElbowYaw', left_elbow_yaw)
    left_elbow_roll = clamp_angle('LElbowRoll', left_elbow_roll)
    right_shoulder_pitch = clamp_angle('RShoulderPitch', right_shoulder_pitch)
    right_shoulder_roll = clamp_angle('RShoulderRoll', right_shoulder_roll)
    right_elbow_yaw = clamp_angle('RElbowYaw', right_elbow_yaw)
    right_elbow_roll = clamp_angle('RElbowRoll', right_elbow_roll)

    angle_lists = [
        left_shoulder_pitch * TO_RAD,
        left_shoulder_roll * TO_RAD,
        left_elbow_yaw * TO_RAD,
        left_elbow_roll * TO_RAD,
        0,
        right_shoulder_pitch * TO_RAD,
        right_shoulder_roll * TO_RAD,
        right_elbow_yaw * TO_RAD,
        right_elbow_roll * TO_RAD
    ]
    motionProxy.setAngles(names, angle_lists, speed)


def get_angles():
    names = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "RShoulderPitch", "RShoulderRoll",
             "RElbowYaw", "RElbowRoll"]
    angles = motionProxy.getAngles(names, True)
    angles_in_degrees = [angle * TO_DEG for angle in angles]
    return dict(zip(names, angles_in_degrees))


def handle_client(client_socket):
    try:
        data = ''
        try:
            while True:
                chunk = client_socket.recv(8192)
                if not chunk:
                    break
                data += chunk

                request_line = data.split('\r\n')[0]
                method, path, _ = request_line.split()

                try:
                    content_length = re.search(r'Content-Length: (\d+)', data)
                    if content_length:
                        body_length = int(content_length.group(1))
                        body = data.split('\r\n\r\n', 1)[1][-body_length:].strip()
                    else:
                        body = ''
                except:
                    pass

                if body_length == 0 or body:
                    break

        except Exception as e:
            pass

        if method == 'POST' and path == '/move_arms':
            data = json.loads(body.strip())
            print(data)
            left_shoulder_pitch = data['left_shoulder_pitch']
            left_shoulder_roll = data['left_shoulder_roll']
            left_elbow_yaw = data['left_elbow_yaw']
            left_elbow_roll = data['left_elbow_roll']
            right_shoulder_pitch = data['right_shoulder_pitch']
            right_shoulder_roll = data['right_shoulder_roll']
            right_elbow_yaw = data['right_elbow_yaw']
            right_elbow_roll = data['right_elbow_roll']

            move_arms(left_shoulder_pitch, left_shoulder_roll, left_elbow_yaw, left_elbow_roll,
                      right_shoulder_pitch, right_shoulder_roll, right_elbow_yaw, right_elbow_roll)

            response = 'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n'
            client_socket.sendall(response.encode('utf-8'))

        elif method == 'GET' and path == '/get_angles':
            angles = get_angles()
            response_body = json.dumps(angles)
            response = 'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}'.format(
                len(response_body), response_body)
            client_socket.sendall(response.encode('utf-8'))

        else:
            response = 'HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n'
            client_socket.sendall(response.encode('utf-8'))

    except Exception as e:
        error_message = 'HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n'
        client_socket.sendall(error_message.encode('utf-8'))
        print(e)
    finally:
        client_socket.close()


def start_server(host, port):
    # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the host and port
    server_socket.bind((host, port))

    # Listen for incoming connections
    server_socket.listen(1)
    print('Socket server started on {}:{}'.format(host, port))

    while True:
        # Accept a client connection
        client_socket, client_address = server_socket.accept()
        print('Connected client:', client_address)

        # Handle the client request
        handle_client(client_socket)

    # Close the server socket
    server_socket.close()


if __name__ == '__main__':
    try:
        motionProxy.setStiffnesses("Body", 1.0)
        move_arms(0, 0, -42, -5,
                  0, 0, 0, 0)
        start_server(host='0.0.0.0', port=5001)
    except Exception as e:
        print("Error"  + str(e))
