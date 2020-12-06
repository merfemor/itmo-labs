cmake_minimum_required(VERSION 3.14)
project(lab2)

set(CMAKE_C_STANDARD 99)

include_directories(/usr/local/Cellar/open-mpi/4.0.5/include)

#set(CMAKE_C_COMPILER gcc-10)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
#add_executable(lab2 main.c)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
add_executable(lab2_mpi mpi_main.c)
