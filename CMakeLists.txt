find_package(Boost COMPONENTS system thread chrono filesystem)

set(ODEINT_ROOT    $ENV{ODEINT_ROOT}   CACHE STRING "odeint root")
set(VEXCL_ROOT     $ENV{VEXCL_ROOT}    CACHE STRING "VexCL root")

MESSAGE(STATUS "Found odeint: ${ODEINT_ROOT}")
MESSAGE(STATUS "Found VexCL: ${VEXCL_ROOT}")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ncwg
    ${ODEINT_ROOT}
    ${VEXCL_ROOT}
    )

find_package(Boost COMPONENTS system filesystem timer)
include_directories(${Boost_INCLUDE_DIRS})
add_definitions(-DBOOST_DISABLE_ASSERTS)

find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(OpenCL QUIET)
if (OPENCL_FOUND)
    include_directories( ${OPENCL_INCLUDE_DIRS} )
endif()

find_package(CUDA)
if (CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "-arch=sm_35")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    include_directories( ${CUDA_INCLUDE_DIRS} )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

option(SHOW_OUTPUT "Show output" OFF)
if (SHOW_OUTPUT)
    add_definitions(-DSHOW_OUTPUT)
endif()

function(add_thrust_cpu_executable target source)
    if (CUDA_FOUND)
        cuda_add_executable(${target} ${source}
            OPTIONS "-Xcompiler;-fopenmp;-O3;-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP;-DCPU_RUN")
        target_link_libraries(${target} gomp ${Boost_LIBRARIES})
    endif()
endfunction()

function(add_thrust_gpu_executable target source)
    if (CUDA_FOUND)
        cuda_add_executable(${target} ${source} OPTIONS "-O3")
        target_link_libraries(${target} ${Boost_LIBRARIES})

    endif()
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${source}
        ${CMAKE_CURRENT_BINARY_DIR}/${source}
        COPYONLY
        )
endfunction()

function(add_vexcl_executable target source)
    if (OPENCL_FOUND)
        add_executable(${target} ${source})
        target_link_libraries(${target} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} )
    endif()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${source}
        ${CMAKE_CURRENT_BINARY_DIR}/${source}
        COPYONLY
        )
endfunction()

function(add_cpu_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} ${Boost_LIBRARIES} )

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${source}
        ${CMAKE_CURRENT_BINARY_DIR}/${source}
        COPYONLY
        )
endfunction()

add_subdirectory(md)
add_subdirectory(lorenz)
add_subdirectory(oscillators)
add_subdirectory(ncwg)
