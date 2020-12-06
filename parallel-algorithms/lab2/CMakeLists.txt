cmake_minimum_required(VERSION 3.14)
project(lab2)

set(CMAKE_C_STANDARD 99)

include_directories(/usr/local/Cellar/open-mpi/4.0.5/include)

#set(CMAKE_C_COMPILER gcc-10)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DPRINT_DEBUG_MATRICES")
#add_executable(lab2 main.c)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DPRINT_DEBUG_MATRICES -DSERIAL_RESULT_CHECK")

add_executable(lab2_mpi mpi_main.c)
