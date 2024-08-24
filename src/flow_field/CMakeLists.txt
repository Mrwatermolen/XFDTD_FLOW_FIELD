aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} XFDTD_FLOW_FIELD_SRC)
add_library(xfdtd_flow_field ${XFDTD_FLOW_FIELD_SRC})
target_link_libraries(xfdtd_flow_field PUBLIC ${XFDTD_FLOW_FIELD_DEPENDENCIES})
target_include_directories(xfdtd_flow_field PRIVATE ${XFDTD_FLOW_FIELD_PRIVATE_INCLUDE_DIR})
