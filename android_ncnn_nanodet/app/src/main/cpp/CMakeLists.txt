project(YOLOv5ss)

cmake_minimum_required(VERSION 3.10.2)

# opencv-mobile-2.4.13.7-android
set(OpenCV_DIR opencv-mobile-2.4.13.7-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

# ncnn-20210525-android-vulkan
set(ncnn_DIR ncnn-20210525-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# 头文件路径
include_directories(.)
include_directories(camera)

add_library(cameraOpencv SHARED camera/ndkcamera.cpp nanodetncnn.cpp nanodet.cpp)
target_link_libraries(cameraOpencv ncnn ${OpenCV_LIBS} camera2ndk mediandk)
