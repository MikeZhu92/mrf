 catkin_add_gtest(${PROJECT_NAME}_gtest test/test_solver.cpp)
 if(TARGET ${PROJECT_NAME}_gtest)
  target_link_libraries(
     ${PROJECT_NAME}_gtest
     ${catkin_LIBRARIES}
   )
 endif()