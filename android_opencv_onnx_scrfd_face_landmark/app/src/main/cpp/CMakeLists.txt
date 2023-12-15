
cmake_minimum_required(VERSION 3.22.1)



project("camera_opencv")


add_library(
             camera_opencv
             SHARED
             native-lib.cpp )

find_library(
              log-lib
              log )

target_link_libraries(
                       camera_opencv
                       ${log-lib} )