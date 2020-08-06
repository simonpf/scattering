set(SHTNS_LIBRARY CACHE "NODOC" INTERNAL)
set(SCATLIB_TEST_PATH ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(utils.py utils.py @ONLY)
configure_file(test_sht.py test_sht.py COPYONLY)
configure_file(test_interpolation.py test_interpolation.py COPYONLY)
configure_file(test_scattering_data.py test_scattering_data.py COPYONLY)
