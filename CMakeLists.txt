cmake_minimum_required(VERSION 3.8)
project(urdf_parser)

find_package(ament_cmake REQUIRED)

ament_python_install_package(${PROJECT_NAME})

ament_package()
