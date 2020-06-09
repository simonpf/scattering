set(SHTNS_LIBRARY CACHE "NODOC" INTERNAL)
add_executable(shtns shtns.cxx)
add_dependencies(shtns libshtns)
target_link_libraries(shtns ${SHTNS_LIBRARY})
