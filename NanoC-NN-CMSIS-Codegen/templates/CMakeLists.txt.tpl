cmake_minimum_required(VERSION 3.16)
project(nanoc_cmsis_generated C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(nanoc_cmsis_generated
    src/main.c
    src/model.c
)

target_include_directories(nanoc_cmsis_generated PRIVATE
    include
)
