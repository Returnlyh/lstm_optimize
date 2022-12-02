#1.cmake verson
cmake_minimum_required(VERSION 3.10)

#2.project name
project(lstm_opt LANGUAGES C) 

#3.set environment variable
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS   "${CMAKE_CXX_FLAGS} -msse -pg")  
set(CMAKE_BUILD_TYPE "Release")

# add_definitions(-DW_TYPE_INT8=1)
# add_definitions(-DFAST_MATH=0)
option(W_TYPE_INT8 "The weight parameter data type uses int8" ON)
option(FAST_MATH "Using fast mathematical approximations" ON)

#4. include
include_directories(${CMAKE_SOURCE_DIR}/include)

#5. source code
aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC_LIST)

#6. lib
link_libraries(m)

#7. do
add_executable(lstm_opt ${SRC_LIST})