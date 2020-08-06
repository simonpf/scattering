import sys
import netCDF4
import os

# Import test utils.
sys.path.append(os.path.dirname(__file__))
import utils

class ScatteringDataAzymuthallyRandom:
    def __init__(self, filename):
        handle = netCDF4.Dataset(filename)
        self.azimuth_angles_incoming = handle["azimuth_angles_incoming"][:].data
        self.zenith_angles_incoming = handle["zenith_angles_incoming"][:].data
        self.azimuth_angles_scattering = handle["azimuth_angles_scattering"][:].data
        self.zenith_angles_scattering = handle["zenith_angles_scattering"][:].data
        self.phase_matrix = handle["phase_matrix"][:].data
        self.extinction_matrix = handle["extinction_matrix"][:].data
        self.absorption_vector = handle["absorption_vector"][:].data

scattering_data = ScatteringDataAzymuthallyRandom(utils.get_azimuthally_random_scattering_data())

