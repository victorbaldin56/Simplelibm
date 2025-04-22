add_executable(logf src/main.c)
target_link_libraries(logf lalogf)

add_test(NAME ulp_test
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test.py)
