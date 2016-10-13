add_vexcl_executable     (lorenz_vexcl_v1  lorenz_vexcl_v1.cpp)
add_vexcl_executable     (lorenz_vexcl_v2  lorenz_vexcl_v2.cpp)
add_vexcl_executable     (lorenz_vexcl_v3  lorenz_vexcl_v3.cpp)

add_thrust_gpu_executable(lorenz_thrust_v1 lorenz_thrust_v1.cu)
add_thrust_cpu_executable(lorenz_cpu_v1    lorenz_thrust_v1.cu)

add_cpu_executable       (lorenz_single    lorenz_single.cpp)
add_cpu_executable       (lorenz_cpu_v2    lorenz_cpu_v2.cpp)

foreach(script run_gpu_tests.sh run_cpu_tests.sh)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${script}
        ${CMAKE_CURRENT_BINARY_DIR}/${script}
        COPYONLY
        )
endforeach()
