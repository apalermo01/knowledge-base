cmake_minimum_required(VERSION 3.4)
project(pybind_tutorial)
add_subdirectory(pybind11)
pybind11_add_module(module_name main.cpp)