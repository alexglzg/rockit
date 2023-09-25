cmake_minimum_required(VERSION 3.0.2)
project(admm_test)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

## System dependencies are found with CMake's conventions
#############################
## Find CasADi
#############################
## -----------
## Option 1
## -----------
# set(CASADI_DIR "/usr/local")
# set(CMAKE_PREFIX_PATH ${CASADI_DIR} ${CMAKE_PREFIX_PATH})
# find_package(casadi 
#   REQUIRED 
#   HINTS ${CASADI_DIR} ${CASADI_DIR}/lib
# )
# set(CASADI_LIBRARY_DIRS ${CASADI_DIR}/lib)
# set(CASADI_INCLUDEDIR ${CASADI_DIR}/include)
# set(CASADI_LIBRARIES casadi)

## -----------
## Option 2
## -----------
## Test if pkg-config is able to find your Ipopt installation by using 'pkg-config --libs ipopt'
set(CASADI_DIR "/usr/local")
set(CMAKE_PREFIX_PATH ${CASADI_DIR} ${CMAKE_PREFIX_PATH})
set(PKG_CONFIG_PATH ${CASADI_DIR}/lib/pkgconfig ${PKG_CONFIG_PATH})
find_package(PkgConfig REQUIRED)
pkg_search_module(CASADI casadi REQUIRED)

# ## -----------

# # if(NOT CASADI_LIBRARIES)
# #   message(SEND_ERROR "Could not find casadi.pc - did you set PKG_CONFIG_PATH?")
# # endif()

# message("CASADI_LIBRARY_DIRS: ${CASADI_LIBRARY_DIRS}")
# message("CASADI_LIBRARIES: ${CASADI_LIBRARIES}")
# message("CASADI_INCLUDEDIR: ${CASADI_INCLUDEDIR}")

#############################
## Find IPOPT
#############################
## Test if pkg-config is able to find your Ipopt installation by using 'pkg-config --libs ipopt'
find_package(PkgConfig)
pkg_search_module(IPOPT ipopt REQUIRED)

#############################
## Find Eigen
#############################
set(EIGEN_INC "/usr/include/eigen3")


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES admm_test
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########
link_directories(${CASADI_LIBRARY_DIRS})

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(admm_test src/admm_test.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(admm_test
  ${CASADI_LIBRARIES}
  ipopt
  ${catkin_LIBRARIES}
)

target_include_directories(admm_test PUBLIC ${CASADI_INCLUDEDIR} ${EIGEN_INC})
