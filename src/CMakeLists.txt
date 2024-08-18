aux_source_directory(. SRC_LIST)
add_executable(flow_field ${SRC_LIST})

target_link_libraries(flow_field PRIVATE xfdtd::xfdtd_core)
target_include_directories(flow_field PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
