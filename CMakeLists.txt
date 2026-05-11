cmake_minimum_required(VERSION 3.15)
project(RegisterAllocation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(solver
        main.cpp
)
target_include_directories(solver PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(generator
        src/generator.cpp
)