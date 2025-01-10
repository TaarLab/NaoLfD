cmake_minimum_required(VERSION 3.0.2)
project(nao_ros_direct)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nao_lfd_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy nao_lfd_msgs
)

catkin_install_python(PROGRAMS
  scripts/nao_direct_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
