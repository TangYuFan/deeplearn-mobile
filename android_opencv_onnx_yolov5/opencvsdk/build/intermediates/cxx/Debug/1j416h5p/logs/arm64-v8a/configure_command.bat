@echo off
"C:\\software\\android_sdk_4onnx\\cmake\\3.18.1\\bin\\cmake.exe" ^
  "-HD:\\work\\workspace\\deeplearn-mobile\\android_opencv_onnx_yolov5\\opencvsdk\\libcxx_helper" ^
  "-DCMAKE_SYSTEM_NAME=Android" ^
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" ^
  "-DCMAKE_SYSTEM_VERSION=21" ^
  "-DANDROID_PLATFORM=android-21" ^
  "-DANDROID_ABI=arm64-v8a" ^
  "-DCMAKE_ANDROID_ARCH_ABI=arm64-v8a" ^
  "-DANDROID_NDK=C:\\software\\android_sdk_4onnx\\ndk\\23.1.7779620" ^
  "-DCMAKE_ANDROID_NDK=C:\\software\\android_sdk_4onnx\\ndk\\23.1.7779620" ^
  "-DCMAKE_TOOLCHAIN_FILE=C:\\software\\android_sdk_4onnx\\ndk\\23.1.7779620\\build\\cmake\\android.toolchain.cmake" ^
  "-DCMAKE_MAKE_PROGRAM=C:\\software\\android_sdk_4onnx\\cmake\\3.18.1\\bin\\ninja.exe" ^
  "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=D:\\work\\workspace\\deeplearn-mobile\\android_opencv_onnx_yolov5\\opencvsdk\\build\\intermediates\\cxx\\Debug\\1j416h5p\\obj\\arm64-v8a" ^
  "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=D:\\work\\workspace\\deeplearn-mobile\\android_opencv_onnx_yolov5\\opencvsdk\\build\\intermediates\\cxx\\Debug\\1j416h5p\\obj\\arm64-v8a" ^
  "-DCMAKE_BUILD_TYPE=Debug" ^
  "-BD:\\work\\workspace\\deeplearn-mobile\\android_opencv_onnx_yolov5\\opencvsdk\\.cxx\\Debug\\1j416h5p\\arm64-v8a" ^
  -GNinja ^
  "-DANDROID_STL=c++_shared"
