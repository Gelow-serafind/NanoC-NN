cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME nanoc-nn)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# CMSIS-NN source set for terminal TDD. Keep this in the project file instead
# of editing third_party/CMSIS-NN.
file(GLOB_RECURSE NANOC_CMSIS_NN_SOURCES
    ${CMAKE_SOURCE_DIR}/../../../third_party/CMSIS-NN/Source/*.c
)
list(FILTER NANOC_CMSIS_NN_SOURCES EXCLUDE REGEX ".*_f16\\.c$")
list(FILTER NANOC_CMSIS_NN_SOURCES EXCLUDE REGEX ".*_f32\\.c$")
list(FILTER NANOC_CMSIS_NN_SOURCES EXCLUDE REGEX ".*/LSTMFunctions/.*")

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user sources here
    app/nanoc_app.c
    app/nanoc_case.c
    app/nanoc_protocol.c
    app/nanoc_time.c
    app/nanoc_uart.c
    app/cases/nanoc_case_dummy_echo.c
    app/cases/nanoc_case_mnist_int8.c
    app/cases/nanoc_case_terminal_model.c
    app/models/terminal_model/src/model.c
    ${NANOC_CMSIS_NN_SOURCES}
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
    app
    app/cases
    app/models/terminal_model/include
    ${CMAKE_SOURCE_DIR}/../../../third_party/CMSIS-NN/Include
    ${CMAKE_SOURCE_DIR}/../../../third_party/CMSIS-NN/Include/Internal
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
    NANOC_ENABLE_CMSIS_NN=1
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx

    # Add user defined libraries
)
