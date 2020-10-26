file(GLOB_RECURSE shtns_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} shtns/*)
foreach(f IN LISTS shtns_files)
  configure_file(${f} ${f} COPYONLY)
endforeach()

set(SHTNS_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/shtns/libshtns.a CACHE "NODOC" INTERNAL FORCE)

add_custom_command(
  OUTPUT ${SHTNS_LIBRARY}
  COMMAND ./configure CFLAGS=-fpic ENABLE_MEM=no
  COMMAND make
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shtns
  )

add_custom_target(libshtns DEPENDS ${SHTNS_LIBRARY})
