cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
project(myproject)

# find_package(OpenCV 4.0 COMPONENTS [comp_name])
# if(OpenCV_[comp_name]_FOUND)
#    add_definitions("-DHAVE_[comp_name]")

find_package(OpenCV REQUIRED dnn highgui imgproc)

add_executable(mobilenet_image dnn_mobilenet.cpp)
add_executable(yolo_image dnn_yolo.cpp)
add_executable(maskrcnn_image dnn_maskrcnn.cpp)

foreach(item mobilenet_image yolo_image maskrcnn_image)
  target_include_directories(${item} PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${item}
    opencv_dnn
    opencv_highgui
    opencv_imgproc
  )
  set(out_directory "$<TARGET_FILE_DIR:${item}>")
  add_custom_command(TARGET ${item} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_SOURCE_DIR}/resources ${out_directory}/resources
  )
endforeach()



