import os

scatlib_test_path = "@SCATLIB_TEST_PATH@"

def get_azimuthally_random_scattering_data():
    return os.path.join(scatlib_test_path, "data", "scattering_data_azimuthally_random.nc")
