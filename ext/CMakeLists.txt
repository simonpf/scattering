file(GLOB_RECURSE shtns_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} shtns/*)
foreach(f IN LISTS shtns_files)
  configure_file(${f} ${f} COPYONLY)
endforeach()

add_custom_command(
  PRE_BUILD
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shtns/libshtns.a
  COMMAND ./configure
  COMMAND make
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shtns
  )
