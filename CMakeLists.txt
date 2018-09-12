cmake_minimum_required( VERSION 2.8 )
project( backend_testing )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( G2O REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( Cholmod )

# CSparse
FIND_PACKAGE( CSparse REQUIRED )

#PCL
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

include_directories( ${EIGEN3_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

ADD_LIBRARY( slambase slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( slam_tum slam_tum.cpp )
TARGET_LINK_LIBRARIES( slam_tum
    slambase
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba ${CHOLMOD_LIBRARIES} ${CSPARSE_LIBRARY})
