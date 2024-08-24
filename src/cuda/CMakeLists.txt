include(CheckLanguage)
check_language(CUDA)

set(XFDTD_FLOW_FIELD_CUDA_ENABLED OFF CACHE BOOL "Enable CUDA support for flow_field")

if (CMAKE_CUDA_COMPILER)
  set(XFDTD_FLOW_FIELD_CUDA_ENABLED ON)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  message(STATUS "CUDA not found, cannot build target flow_field_cuda")
endif()

if (XFDTD_FLOW_FIELD_CUDA_ENABLED)
  enable_language(CUDA)
  find_package(xfdtd_cuda REQUIRED)
  aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} XFDTD_FLOW_FIELD_CUDA_SRC_LIST)
  add_executable(flow_field_cuda ${XFDTD_FLOW_FIELD_CUDA_SRC_LIST})
  target_link_libraries(flow_field_cuda PRIVATE 
    ${XFDTD_FLOW_FIELD_DEPENDENCIES} 
    xfdtd::xfdtd_cuda 
    xfdtd_flow_field)
  target_include_directories(flow_field_cuda PRIVATE ${XFDTD_FLOW_FIELD_PRIVATE_INCLUDE_DIR})
endif()
