#!/usr/bin/env python
"""ROS1 node: services + closed-loop topics via NAOqi ALProxy."""

from __future__ import print_function

import rospy

from naoqi import ALProxy
import almath

from nao_lfd_msgs.msg import ArmJoints
from nao_lfd_msgs.srv import MoveArms, MoveArmsResponse, GetAngles, GetAnglesResponse


JOINT_LIMITS = {
    "LElbowYaw": {"min": -119.5, "max": 119.5},
    "LElbowRoll": {"min": -88.5, "max": -2},
    "LShoulderRoll": {"min": -18, "max": 76},
    "LShoulderPitch": {"min": -119.5, "max": 119.5},
    "RElbowYaw": {"min": -119.5, "max": 119.5},
    "RElbowRoll": {"min": 2, "max": 88.5},
    "RShoulderRoll": {"min": -76, "max": 18},
    "RShoulderPitch": {"min": -119.5, "max": 119.5},
}

ARM_JOINT_NAMES = [
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
]

READ_JOINT_NAMES = [
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
]


def clamp_angle(joint_name, angle):
    limits = JOINT_LIMITS[joint_name]
    return max(limits["min"], min(limits["max"], angle))


class NaoDirectNode(object):
    def __init__(self):
        nao_ip = rospy.get_param("~nao_ip", "192.168.101.146")
        nao_port = int(rospy.get_param("~nao_port", 9559))
        self.move_speed = float(rospy.get_param("~move_speed", 0.05))
        rate_hz = float(rospy.get_param("~joint_state_rate", 10.0))
        set_stiffness = bool(rospy.get_param("~set_stiffness_on_start", False))
        home_on_start = bool(rospy.get_param("~home_on_start", False))

        self.motion = ALProxy("ALMotion", nao_ip, nao_port)
        self.to_rad = almath.TO_RAD
        self.to_deg = almath.TO_DEG

        if set_stiffness:
            self.motion.setStiffnesses("Body", 1.0)
        if home_on_start:
            self._move_arms_values(0.0, 0.0, -42.0, -5.0, 0.0, 0.0, 0.0, 0.0)

        self.joint_pub = rospy.Publisher("/nao/joint_states", ArmJoints, queue_size=10)
        rospy.Subscriber("/nao/joint_command", ArmJoints, self._joint_command_cb, queue_size=10)

        self.move_srv = rospy.Service("/nao/move_arms", MoveArms, self._move_arms_cb)
        self.get_srv = rospy.Service("/nao/get_angles", GetAngles, self._get_angles_cb)

        self._timer = rospy.Timer(rospy.Duration(1.0 / max(rate_hz, 0.1)), self._publish_joint_states)
        rospy.loginfo("nao_direct ready (ALProxy %s:%s)", nao_ip, nao_port)

    def _move_arms_values(
        self,
        left_shoulder_pitch,
        left_shoulder_roll,
        left_elbow_yaw,
        left_elbow_roll,
        right_shoulder_pitch,
        right_shoulder_roll,
        right_elbow_yaw,
        right_elbow_roll,
    ):
        self.motion.setStiffnesses("Body", 1.0)
        angle_lists = [
            clamp_angle("LShoulderPitch", left_shoulder_pitch) * self.to_rad,
            clamp_angle("LShoulderRoll", left_shoulder_roll) * self.to_rad,
            clamp_angle("LElbowYaw", left_elbow_yaw) * self.to_rad,
            clamp_angle("LElbowRoll", left_elbow_roll) * self.to_rad,
            0.0,
            clamp_angle("RShoulderPitch", right_shoulder_pitch) * self.to_rad,
            clamp_angle("RShoulderRoll", right_shoulder_roll) * self.to_rad,
            clamp_angle("RElbowYaw", right_elbow_yaw) * self.to_rad,
            clamp_angle("RElbowRoll", right_elbow_roll) * self.to_rad,
        ]
        self.motion.setAngles(ARM_JOINT_NAMES, angle_lists, self.move_speed)

    def _move_from_msg(self, joints):
        self._move_arms_values(
            joints.left_shoulder_pitch,
            joints.left_shoulder_roll,
            joints.left_elbow_yaw,
            joints.left_elbow_roll,
            joints.right_shoulder_pitch,
            joints.right_shoulder_roll,
            joints.right_elbow_yaw,
            joints.right_elbow_roll,
        )

    def _read_joints(self):
        angles = self.motion.getAngles(READ_JOINT_NAMES, True)
        degrees = [angle * self.to_deg for angle in angles]
        msg = ArmJoints()
        (
            msg.left_shoulder_pitch,
            msg.left_shoulder_roll,
            msg.left_elbow_yaw,
            msg.left_elbow_roll,
            msg.right_shoulder_pitch,
            msg.right_shoulder_roll,
            msg.right_elbow_yaw,
            msg.right_elbow_roll,
        ) = degrees
        return msg

    def _move_arms_cb(self, req):
        try:
            self._move_from_msg(req.joints)
            return MoveArmsResponse(success=True, message="ok")
        except Exception as exc:
            rospy.logerr("move_arms failed: %s", exc)
            return MoveArmsResponse(success=False, message=str(exc))

    def _get_angles_cb(self, _req):
        try:
            joints = self._read_joints()
            return GetAnglesResponse(joints=joints, success=True, message="ok")
        except Exception as exc:
            rospy.logerr("get_angles failed: %s", exc)
            return GetAnglesResponse(joints=ArmJoints(), success=False, message=str(exc))

    def _joint_command_cb(self, msg):
        try:
            self._move_from_msg(msg)
        except Exception as exc:
            rospy.logerr("joint_command failed: %s", exc)

    def _publish_joint_states(self, _event):
        try:
            self.joint_pub.publish(self._read_joints())
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "joint_states publish failed: %s", exc)


def main():
    rospy.init_node("nao_direct")
    NaoDirectNode()
    rospy.spin()


if __name__ == "__main__":
    main()
