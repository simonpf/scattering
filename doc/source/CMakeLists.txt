file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")
foreach(file ${files})
  configure_file(${file} ${file} @ONLY)
endforeach()

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.rst")
foreach(file ${files})
  configure_file(${file} ${file} @ONLY)
endforeach()
