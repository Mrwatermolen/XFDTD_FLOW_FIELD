set(XFDTD_FLOW_FIELD_PRIVATE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(cuda)
add_subdirectory(flow_field)

aux_source_directory(. SRC_LIST)
add_executable(flow_field_cpu ${SRC_LIST})

target_link_libraries(flow_field_cpu PRIVATE xfdtd::xfdtd_core xfdtd_flow_field)
target_include_directories(flow_field_cpu PRIVATE ${XFDTD_FLOW_FIELD_PRIVATE_INCLUDE_DIR})
