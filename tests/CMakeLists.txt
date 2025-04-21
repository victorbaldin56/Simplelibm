add_executable(logf src/main.c)
target_link_libraries(logf lalogf)

add_test(NAME logf_test
         COMMAND python3 ./test.py)
