cmake_minimum_required(VERSION 3.8)
project(agri_ugv_four_ws)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

find_package(ament_cmake REQUIRED)

# The four-wheel-steering core: kinematics, speed limits, steer first; plain C++, no ROS
add_library(four_ws_core SHARED
  src/kinematics.cpp
  src/velocity_limiter.cpp
  src/steering.cpp
)
target_include_directories(four_ws_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS four_ws_core
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_four_ws
    test/test_kinematics.cpp
    test/test_velocity_limiter.cpp
    test/test_steering.cpp
  )
  target_link_libraries(test_four_ws four_ws_core)
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_package()
