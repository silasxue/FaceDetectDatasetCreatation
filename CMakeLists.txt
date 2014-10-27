cmake_minimum_required(VERSION 2.6)
project(det_face)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
add_executable( detFaceInVideo video_main.cpp)


include_directories( /media/yuanyang/disk1/libs/dlib-18.10/ )
include_directories( ${Boost_INCLUDE_DIRS} )
set( dlibo "-lX11 -lz -lpthread")

add_library(picort picort.h picort.cpp)
add_library(faceregion faceRegion.hpp)
add_library(facefilter FaceFilter.cpp FaceFilter.h)
add_library(facedetwithfilter FacedetWithFilter.cpp FacedetWithFilter.h)
add_library(dlibsource /media/yuanyang/disk1/libs/dlib-18.10/dlib/all/source.cpp)

target_link_libraries( detFaceInVideo facedetwithfilter dlibsource  ${OpenCV_LIBS} ${Boost_LIBRARIES} ${dlibo})
target_link_libraries( facedetwithfilter facefilter picort ${OpenCV_LIBS})
target_link_libraries( faceregion ${OpenCV_LIBS})
target_link_libraries( facefilter faceregion ${OpenCV_LIBS} )

