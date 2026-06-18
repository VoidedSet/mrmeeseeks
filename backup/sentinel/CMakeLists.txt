cmake_minimum_required(VERSION 3.10)
project(sentinel-daemon CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Find libudev
find_library(UDEV_LIB udev REQUIRED)

# Source files
set(SOURCES
    src/main.cpp
    src/collectors/inotify_collector.cpp
    src/collectors/process_collector.cpp
    src/collectors/network_collector.cpp
    src/collectors/device_collector.cpp
)

add_executable(sentinel-daemon ${SOURCES})

# Link threads and libudev
target_link_libraries(sentinel-daemon PRIVATE pthread ${UDEV_LIB})
