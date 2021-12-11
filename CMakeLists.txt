cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(myproject)

# find_package(OpenCV 4.0 COMPONENTS [comp_name])
# if(OpenCV_[comp_name]_FOUND)
#    add_definitions("-DHAVE_[comp_name]")

find_package(OpenCV REQUIRED dnn highgui imgproc)
add_executable(run main.cpp)

target_include_directories(run PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(run
  opencv_dnn
  opencv_highgui
  opencv_imgproc
)

set(out_directory $<TARGET_FILE_DIR:run>)

add_custom_command(TARGET run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/resources ${out_directory}/resources
)

