#!/usr/bin/env python
"""ROS1 bridge: services + closed-loop topics via existing nao_server HTTP API."""

from __future__ import print_function

import rospy
import requests

from nao_lfd_msgs.msg import ArmJoints
from nao_lfd_msgs.srv import MoveArms, MoveArmsResponse, GetAngles, GetAnglesResponse


class NaoHttpBridge(object):
    def __init__(self):
        self.base_url = rospy.get_param("~nao_http_url", "http://127.0.0.1:5001").rstrip("/")
        self.timeout = rospy.get_param("~http_timeout", 5.0)
        rate_hz = float(rospy.get_param("~joint_state_rate", 10.0))

        self.joint_pub = rospy.Publisher("/nao/joint_states", ArmJoints, queue_size=10)
        rospy.Subscriber("/nao/joint_command", ArmJoints, self._joint_command_cb, queue_size=10)

        self.move_srv = rospy.Service("/nao/move_arms", MoveArms, self._move_arms_cb)
        self.get_srv = rospy.Service("/nao/get_angles", GetAngles, self._get_angles_cb)

        self._timer = rospy.Timer(rospy.Duration(1.0 / max(rate_hz, 0.1)), self._publish_joint_states)
        rospy.loginfo("nao_http_bridge ready (HTTP %s)", self.base_url)

    def _joints_to_dict(self, joints):
        return {
            "left_shoulder_pitch": joints.left_shoulder_pitch,
            "left_shoulder_roll": joints.left_shoulder_roll,
            "left_elbow_yaw": joints.left_elbow_yaw,
            "left_elbow_roll": joints.left_elbow_roll,
            "right_shoulder_pitch": joints.right_shoulder_pitch,
            "right_shoulder_roll": joints.right_shoulder_roll,
            "right_elbow_yaw": joints.right_elbow_yaw,
            "right_elbow_roll": joints.right_elbow_roll,
        }

    def _dict_to_joints(self, data):
        msg = ArmJoints()
        msg.left_shoulder_pitch = float(data.get("LShoulderPitch", data.get("left_shoulder_pitch", 0.0)))
        msg.left_shoulder_roll = float(data.get("LShoulderRoll", data.get("left_shoulder_roll", 0.0)))
        msg.left_elbow_yaw = float(data.get("LElbowYaw", data.get("left_elbow_yaw", 0.0)))
        msg.left_elbow_roll = float(data.get("LElbowRoll", data.get("left_elbow_roll", 0.0)))
        msg.right_shoulder_pitch = float(data.get("RShoulderPitch", data.get("right_shoulder_pitch", 0.0)))
        msg.right_shoulder_roll = float(data.get("RShoulderRoll", data.get("right_shoulder_roll", 0.0)))
        msg.right_elbow_yaw = float(data.get("RElbowYaw", data.get("right_elbow_yaw", 0.0)))
        msg.right_elbow_roll = float(data.get("RElbowRoll", data.get("right_elbow_roll", 0.0)))
        return msg

    def _post_move_arms(self, joints):
        url = "{}/move_arms".format(self.base_url)
        response = requests.post(url, json=self._joints_to_dict(joints), timeout=self.timeout)
        response.raise_for_status()

    def _get_angles_http(self):
        url = "{}/get_angles".format(self.base_url)
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return self._dict_to_joints(response.json())

    def _move_arms_cb(self, req):
        try:
            self._post_move_arms(req.joints)
            return MoveArmsResponse(success=True, message="ok")
        except Exception as exc:
            rospy.logerr("move_arms failed: %s", exc)
            return MoveArmsResponse(success=False, message=str(exc))

    def _get_angles_cb(self, _req):
        try:
            joints = self._get_angles_http()
            return GetAnglesResponse(joints=joints, success=True, message="ok")
        except Exception as exc:
            rospy.logerr("get_angles failed: %s", exc)
            return GetAnglesResponse(joints=ArmJoints(), success=False, message=str(exc))

    def _joint_command_cb(self, msg):
        try:
            self._post_move_arms(msg)
        except Exception as exc:
            rospy.logerr("joint_command failed: %s", exc)

    def _publish_joint_states(self, _event):
        try:
            self.joint_pub.publish(self._get_angles_http())
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "joint_states publish failed: %s", exc)


def main():
    rospy.init_node("nao_http_bridge")
    NaoHttpBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
