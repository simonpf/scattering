add_subdirectory(scatlib)

#
# Version from git
#

execute_process(
  COMMAND git describe --tags --abbrev=0
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE VERSION
  )

string(REGEX REPLACE "\n$" "" VERSION "${VERSION}")
configure_file(setup.py setup.py @ONLY)
configure_file(scatlib/__init__.py scatlib/__init__.py @ONLY)
