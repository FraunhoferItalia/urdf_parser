cmake_minimum_required(VERSION 3.8)
project(urdf_parser)

find_package(catkin REQUIRED COMPONENTS
)
find_package(xbot2 REQUIRED)

catkin_python_setup()

catkin_package()
