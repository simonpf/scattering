"""
Test utils providing acces to test data.
"""
import os

SCATLIB_TEST_PATH = "@SCATLIB_TEST_PATH@"

def get_data_azimuthally_random():
    return os.path.join(SCATLIB_TEST_PATH, "data", "scattering_data_azimuthally_random.nc")
