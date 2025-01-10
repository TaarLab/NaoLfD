import math
import socket
import json
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()

joint_handles = [sim.getObject(f'/{joint_name}') for joint_name in [
    'LElbowYaw', 'LElbowRoll', 'LShoulderRoll', 'LShoulderPitch',
    'RElbowYaw', 'RElbowRoll', 'RShoulderRoll', 'RShoulderPitch'
]]

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

def update_robot_joints(new_angles):
    global sim, joint_handles, joint_limits
    if sim is not None and joint_handles is not None:
        for joint, angle, joint_name in zip(joint_handles, new_angles, joint_limits.keys()):
            min_angle = joint_limits[joint_name]['min']
            max_angle = joint_limits[joint_name]['max']
            angle = min(max(angle, min_angle), max_angle)  # Ensure angle is within limits
            sim.setJointTargetPosition(joint, math.radians(angle))


def handle_client(client_socket):
    try:
        data = b''
        while True:
            chunk = client_socket.recv(8192)
            if not chunk:
                break
            data += chunk

            if b'\r\n\r\n' in data[:len(data) - 2]:
                break

        if data:
            request_lines = data.decode().split('\r\n')
            method, path, _ = request_lines[0].split()

            if method == 'POST':
                for line in request_lines:
                    if line.startswith('Content-Length:'):
                        content_length = int(line.split(':')[1].strip())
                        body = data.decode().split('\r\n\r\n', 1)[1][:content_length].strip()

                if body:
                    try:
                        data = json.loads(body)
                        angles = [
                            data['Left_Elbow_Yaw'],
                            data['Left_Elbow_Roll'],
                            data['Left_Shoulder_Roll'],
                            data['Left_Shoulder_Pitch'],
                            data['Right_Elbow_Yaw'],
                            data['Right_Elbow_Roll'],
                            data['Right_Shoulder_Roll'],
                            data['Right_Shoulder_Pitch']
                        ]
                        update_robot_joints(angles)

                        response = 'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n'
                        client_socket.sendall(response.encode('utf-8'))
                    except Exception as e:
                        print(e)
                        response = 'HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n'
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
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print('Socket server started on {}:{}'.format(host, port))

    while True:
        client_socket, client_address = server_socket.accept()
        print('Connected client:', client_address)
        handle_client(client_socket)

if __name__ == '__main__':
    start_server(host='0.0.0.0', port=5188)
