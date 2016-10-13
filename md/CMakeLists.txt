add_cpu_executable       (mdc_cpu_v1    mdc_cpu_v1.cpp)
add_cpu_executable       (mdc_cpu_v2    mdc_cpu_v2.cpp)

add_vexcl_executable     (mdc_vexcl_v1  mdc_vexcl_v1.cpp)
add_vexcl_executable     (mdc_vexcl_v2  mdc_vexcl_v2.cpp)

add_thrust_gpu_executable(mdc_thrust_v1 mdc_thrust_v1.cu)
add_thrust_gpu_executable(mdc_thrust_v2 mdc_thrust_v2.cu)

foreach(script run_gpu_tests.sh run_cpu_tests.sh)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${script}
        ${CMAKE_CURRENT_BINARY_DIR}/${script}
        COPYONLY
        )
endforeach()
