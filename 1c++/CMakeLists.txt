# Authors: Frerk Saxen and Philipp Werner
# License: BSD 2-Clause "Simplified" License (see LICENSE file in root directory)

cmake_minimum_required (VERSION 2.6)

project (ICCV17Challenge)

set(include_dirs "${PROJECT_SOURCE_DIR}/include")

# Add dlib to project
set(dlib_path "" CACHE PATH "Dlib folder")
include(${dlib_path}/dlib/cmake)

message(WARNING "Please make sure that dlib found CUDA and cudnn. Otherwise this project will not work.")
# The message should look something like:
# "Found CUDA: /usr/local/cuda (found suitable version 8.0, minimum required is 7.5)"
# "Looking for cuDNN install..."
# "Building a CUDA test project to see if your compiler is compatible with CUDA..."
# "Checking if you have the right version of cuDNN installed."
# "Found cuDNN: /usr/lib/x86_64-linux-gnu/libcudnn.so"

# Add OpenCV to project
if(NOT OpenCV_FOUND)
	FIND_PACKAGE(OpenCV REQUIRED)
	
	# Problem:    2.4.8 (and maybe other version) set OpenCV_FOUND as a CACHE variable
	# Effect:     find_package is only called during the first run of configure
	# Workaround: set OpenCV_FOUND as regular variable (as usual)
	if(OpenCV_FOUND)
		unset(OpenCV_FOUND CACHE)
		unset(OPENCV_FOUND CACHE)
		set(OpenCV_FOUND true)
	else()
		unset(OpenCV_FOUND CACHE)
		unset(OPENCV_FOUND CACHE)
	endif()
endif()

# add opencv library to project
if(NOT OpenCV_FOUND)
	message(ERROR "OpenCV was not found.")
endif()
set(include_dirs ${include_dirs} ${OpenCV_INCLUDE_DIRS})

set(src_dir "${PROJECT_SOURCE_DIR}/src")
file(GLOB src_files ${src_dir}/*.cpp)

#message(STATUS "include_dirs: " ${include_dirs})
#message(STATUS "src_dir: " ${src_dir})
#message(STATUS "src_files: " ${src_files})

include_directories("${include_dirs}")
add_executable(ICCV17Challenge ${src_files})
target_link_libraries(ICCV17Challenge dlib::dlib ${OpenCV_LIBS} "stdc++fs")
