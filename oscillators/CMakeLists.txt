add_vexcl_executable(po_vexcl po_vexcl.cpp)
add_thrust_gpu_executable(po_thrust po_thrust.cu)
add_thrust_cpu_executable(po_cpu po_thrust.cu)

foreach(script run_gpu_tests.sh run_cpu_tests.sh)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${script}
        ${CMAKE_CURRENT_BINARY_DIR}/${script}
        COPYONLY
        )
endforeach()
