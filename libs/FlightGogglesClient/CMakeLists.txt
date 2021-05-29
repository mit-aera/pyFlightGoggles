CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)

set(FOUND_FlightGogglesClient True)

# OpenCV
find_package(OpenCV REQUIRED)

# ZeroMQ
find_package(PkgConfig)
pkg_check_modules(PC_ZeroMQ QUIET zmq)
find_path(ZeroMQ_INCLUDE_DIR
    NAMES zmq.hpp
    PATHS ${PC_ZeroMQ_INCLUDE_DIRS}
)
find_library(ZeroMQ_LIBRARY
    NAMES zmq
    PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
)

# Add FlightGogglesClient as library
add_library(FlightGogglesClient SHARED FlightGogglesClient.cpp FlightGogglesClient.hpp)

# Link in needed libraries
target_link_libraries(FlightGogglesClient zmq zmqpp ${OpenCV_LIBS} pthread)

# # Expose as library
# target_include_directories(FlightGogglesClient PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
