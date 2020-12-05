cmake_minimum_required(VERSION 3.14)
project(lab2)

set(CMAKE_CXX_STANDARD 98)

set(CMAKE_CXX_COMPILER g++-10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_executable(lab2 main.cpp)

#set(CMAKE_CXX_COMPILER mpic++)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
#add_executable(lab2_mpi mpi_main.cpp)

# mpic++ mpi_main.cpp -o lab2_mpi
# mpirun lab2_mpi