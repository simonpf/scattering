"""
Reference interface for ARTS scattering data base.

This module provides a reference interface to the ARTS SSDB used
to test the C++ implementation.
"""
import numpy as np
import glob
import re
import netCDF4
import bisect
import xarray

RANDOM = "Random"
AZIMUTHALLY_RANDOM = "Azimuthally Random"


class ScatteringData:
    """
    The ScatteringData class holds the scattering data of a particle for
    a specific frequency and temperature. The data thus depends only
    on the incoming and outgoing angles.
    """
    def __init__(self,
                 lon_inc,
                 lat_inc,
                 lon_scat,
                 lat_scat,
                 phase_matrix_data,
                 extinction_matrix_data,
                 absorption_vector_data):
        self._lon_inc = lon_inc
        self._lat_inc = lat_inc
        self._lon_scat = lon_scat
        self._lat_scat = lat_scat
        self._phase_matrix_data = phase_matrix_data
        self._extinction_matrix_data = extinction_matrix_data
        self._absorption_vector_data = absorption_vector_data

        if self._phase_matrix_data.shape[0] == 6:
            self._type = RANDOM
        else:
            self._type = AZIMUTHALLY_RANDOM

    @property
    def lon_inc(self):
        return self._lon_inc

    @property
    def lat_inc(self):
        return self._lat_inc

    @property
    def lon_scat(self):
        return self._lon_scat

    @property
    def lat_scat(self):
        return self._lat_scat

    @property
    def phase_matrix(self):
        if self._type == RANDOM:
            data = self._phase_matrix_data.transpose([1, 2, 3, 4, 0])
            return data.reshape((1, 1,) + data.shape)
        else:
            data = self._phase_matrix_data.transpose([1, 2, 3, 0])
            return data.reshape((1, 1,) + data.shape)

    @property
    def extinction_matrix(self):
        if self._type == RANDOM:
            data = self._extinction_matrix_data.transpose([1, 2, 0])
            return data.reshape((1, 1,) + data.shape[:2] + (1, 1) + data.shape[-1:])
        else:
            data = self._extinction_matrix_data.transpose([1, 2, 0]) * (1.0 + 0.0j)
            return data.reshape((1, 1,) + data.shape[:2] + (1,) + data.shape[-1:])

    @property
    def absorption_vector(self):
        if self._type == RANDOM:
            data = self._absorption_vector_data.transpose([1, 2, 0])
            return data.reshape((1, 1,) + data.shape[:2] + (1, 1) + data.shape[-1:])
        else:
            data = self._absorption_vector_data.transpose([1, 2, 0]) * (1.0 + 0.0j)
            return data.reshape((1, 1,) + data.shape[:2] + (1,) + data.shape[-1:])

    @property
    def backward_scattering_coeff(self):
        if self._type == RANDOM:
            data = self.phase_matrix[:, :, :, :, 0, -1, 0]
            data = data.reshape(data.shape + (1, 1, 1))
        else:
            data = self.phase_matrix[:, :, :, :, -1, 0]
            data = data.reshape(data.shape + (1, 1))
        return data

    @property
    def forward_scattering_coeff(self):
        if self._type == RANDOM:
            data = self.phase_matrix[:, :, :, :, 0, 0, 0]
            data = data.reshape(data.shape + (1, 1, 1))
        else:
            data = self.phase_matrix[:, :, :, :, 0, 0]
            data = data.reshape(data.shape + (1, 1))
        return data


class ParticleFile:
    """
    The particle class represents scattering data for a specific particle. Its
    purpose is to give access to the scattering properties at the different frequencies
    and temperatures that are available.
    """
    def _parse_temperatures_and_frequencies(self):
        group_names = self.file_handle.groups.keys()
        temps = []
        freqs = []
        for gn in group_names:
            match = re.match('Freq([0-9\.]*)GHz_T([0-9\.]*)K', gn)
            freq = match.group(1)
            temp = match.group(2)
            temps.append(temp)
            freqs.append(freq)
        self.temperatures = np.array(temps, dtype=np.float)
        self.frequencies = np.array(freqs, dtype=np.float)

        key = lambda i: (freqs[i], temps[i])
        indices = list(range(self.frequencies.size))
        indices.sort(key=key)

        self.frequencies = self.frequencies[indices]
        self.temperatures = self.temperatures[indices]
        self.frequencies = list(set(self.frequencies))
        self.frequencies.sort()
        self.temperatures = list(set(self.temperatures))
        self.temperatures.sort()
        self.keys = [(f, t) for f in self.frequencies for t in self.temperatures]

    def __init__(self, filename):
        """
        Create a particle object by reading a NetCDF4 file from the ARTS SSDB.
        Args:
            filename: The path of the NetCDF4 file containign the data.
        """
        self.filename = filename
        self.file_handle = netCDF4.Dataset(filename)
        self._parse_temperatures_and_frequencies()

    def __len__(self):
        """
        Returns:
            The number of available unique frequency and temperature
            keys.
        """
        return len(self.frequencies) * len(self.temperatures)

    def __getitem__(self, *args):
        """
        Access scattering data for given frequency and temperature index.
        """
        f, t = self.keys[args[0]]
        return self.get_scattering_data(f, t)

    def get_scattering_data(self, frequency, temperature):
        """
        Return scattering data for given frequency and temperature.


        """
        requested = (frequency, temperature)
        index = bisect.bisect_left(self.keys, requested)
        found = self.keys[index]


        if not (np.all(np.isclose(requested, found))):
            raise Exception("Could not find scattering data for given temperature and frequency")

        group_name = list(self.file_handle.groups.keys())[index]
        group = self.file_handle[group_name]["SingleScatteringData"]

        #
        # Angular grids
        #

        if "aa_inc" in group.variables:
            lon_inc = group["aa_inc"][:]
        else:
            lon_inc = np.zeros(1)

        if "za_inc" in group.variables:
            lat_inc = group["za_inc"][:]
        else:
            lat_inc = np.zeros(1)

        if "aa_scat" in group.variables:
            lon_scat = group["aa_scat"][:]
        else:
            lon_scat = np.zeros(1)

        if "za_scat" in group.variables:
            lat_scat = group["za_scat"][:]
        else:
            lat_scat = np.zeros(1)

        #
        # Data
        #

        if "phaMat_data" in group.variables:
            phase_matrix_data = group["phaMat_data"][:]
        else:
            phase_matrix_data = (group["phaMat_data_real"][:]
                                 + 1j * group["phaMat_data_imag"][:])

        if "extMat_data" in group.variables:
            extinction_matrix_data = group["extMat_data"][:]
        else:
            extinction_matrix_data = (group["extMat_data_real"][:]
                                 + 1j * group["extMat_data_imag"][:])

        if "absVec_data" in group.variables:
            absorption_vector_data = group["absVec_data"][:]
        else:
            absorption_vector_data = (group["absVec_data_real"][:]
                                      + 1j * group["absVec_data_imag"][:])

        return ScatteringData(lon_inc,
                              lat_inc,
                              lon_scat,
                              lat_scat,
                              phase_matrix_data,
                              extinction_matrix_data,
                              absorption_vector_data)


class Habit:

    @staticmethod
    def get_particle_props(path):
        filename = os.path.basename(path)
        match = re.match('Dmax([0-9]*)um_Dveq([0-9]*)um_Mass([-0-9\.e]*)kg\.nc', filename)
        dmax = np.float32(match.group(1)) * 1e-6
        dveq = np.float32(match.group(2)) * 1e-6
        mass = np.float32(match.group(3))
        return (dmax, dveq, mass)

    def __init__(self,
                 name,
                 phase,
                 kind,
                 riming,
                 orientation,
                 path):
        self._name = name
        self._phase = phase
        self._kind = kind
        self._riming = riming
        self._orientation = orientation

        self.files = glob.glob(os.path.join(path, "Dmax*Dveq*Mass*.nc"))
        properties = np.array([Habit.get_particle_props(f) for f in self.files])

        self.d_eq = properties[:, 0]
        self.d_max = properties[:, 1]
        self.mass = properties[:, 2]

        indices = np.argsort(self.d_eq)
        self.d_eq = self.d_eq[indices]
        self.d_max = self.d_max[indices]
        self.mass = self.mass[indices]
        self.files = [self.files[i] for i in indices]

    @property
    def name(self):
        return self._name


    def __repr__(self):
        s = f"SSDB Particle: {self._name}\n"
        s += f"\t Phase: {self._phase}, Kind: {self._kind}, "
        s += f"Riming: {self._riming}, Orientation: {self._orientation}"
        return s

def scattering_angles(lat_inc, lon_scat, lat_scat):
    lat_inc, lon_scat, lat_scat = np.meshgrid(lat_inc, lon_scat, lat_scat, indexing="ij")
    cos_theta = np.cos(lat_inc) * np.cos(lat_scat) + np.sin(lat_inc) * np.sin(lat_scat) * np.cos(lon_scat)
    return np.arccos(cos_theta)

def save_arccos(y):
    x = np.arccos(y)
    inds_0 = np.isnan(x) * (y > 0.0)
    x[inds_0] = 0.0
    inds_pi = np.isnan(x) * (y < 0.0)
    x[inds_pi] = np.pi
    return x


def expand_compact_format(lat_inc, lon_scat, lat_scat, scat_angles, phase_matrix_compact):

    phase_matrix_expanded = np.zeros(phase_matrix_compact.shape[:-1] + (4, 4))
    cos_scat_angles = np.cos(lat_inc) * np.cos(lat_scat) + np.sin(lat_inc) * np.sin(lat_scat) * np.cos(lon_scat)

    cos_sigma_1 = (np.cos(lat_scat) - np.cos(lat_inc) * cos_scat_angles) / (np.sin(lat_inc) * np.sin(scat_angles))
    cos_sigma_2 = (np.cos(lat_inc) - np.cos(lat_scat) * cos_scat_angles) / (np.sin(lat_scat) * np.sin(scat_angles))
    sigma_1 = save_arccos(cos_sigma_1)
    sigma_2 = save_arccos(cos_sigma_2)

    inds = np.where(np.logical_or(np.isclose(lat_inc, 0.0), np.isclose(lat_inc, np.pi)))
    sigma_1[inds] = lon_scat[inds]
    sigma_2[inds] = 0.0

    inds = np.where(np.logical_or(np.isclose(lat_scat, 0.0), np.isclose(lat_scat, np.pi)))
    sigma_1[inds] = 0.0
    sigma_2[inds] = lon_scat[inds]

    c_1 = np.cos(2.0 * sigma_1).reshape((1, 1, 1) + sigma_1.shape)
    c_2 = np.cos(2.0 * sigma_2).reshape((1, 1, 1) + sigma_1.shape)
    s_1 = np.sin(2.0 * sigma_1).reshape((1, 1, 1) + sigma_1.shape)
    s_2 = np.sin(2.0 * sigma_2).reshape((1, 1, 1) + sigma_1.shape)

    f11 = phase_matrix_compact[:, :, :, :, :, :, 0]
    f12 = phase_matrix_compact[:, :, :, :, :, :, 1]
    f22 = phase_matrix_compact[:, :, :, :, :, :, 2]
    f33 = phase_matrix_compact[:, :, :, :, :, :, 3]
    f34 = phase_matrix_compact[:, :, :, :, :, :, 4]
    f44 = phase_matrix_compact[:, :, :, :, :, :, 5]

    phase_matrix_expanded[:, :, :, :, :, :, 0, 0] = f11

    phase_matrix_expanded[:, :, :, :, :, :, 0, 1] = c_1 * f12
    phase_matrix_expanded[:, :, :, :, :, :, 1, 0] = c_2 * f12
    phase_matrix_expanded[:, :, :, :, :, :, 1, 1] = c_1 * c_2 * f22 - s_1 * s_2 * f33

    phase_matrix_expanded[:, :, :, :, :, :, 0, 2] = s_1 * f12
    phase_matrix_expanded[:, :, :, :, :, :, 1, 2] = s_1 * c_2 * f22 + c_1 * s_2 * f33
    phase_matrix_expanded[:, :, :, :, :, :, 2, 2] = -s_1 * s_2 * f22 + c_1 * c_2 * f33
    phase_matrix_expanded[:, :, :, :, :, :, 2, 1] = -c_1 * s_2 * f22 - s_1 * c_2 * f33
    phase_matrix_expanded[:, :, :, :, :, :, 2, 0] = -s_2 * f12

    phase_matrix_expanded[:, :, :, :, :, :, 0, 3] = 0.0
    phase_matrix_expanded[:, :, :, :, :, :, 1, 3] = s_2 * f34
    phase_matrix_expanded[:, :, :, :, :, :, 2, 3] = c_2 * f34
    phase_matrix_expanded[:, :, :, :, :, :, 3, 3] = f44
    phase_matrix_expanded[:, :, :, :, :, :, 3, 2] = -c_1 * f34
    phase_matrix_expanded[:, :, :, :, :, :, 3, 1] = s_1 * f34
    phase_matrix_expanded[:, :, :, :, :, :, 3, 0] = 0.0

    inds = np.where(lon_scat > np.pi)
    phase_matrix_expanded[:, :, :, inds[0], inds[1], inds[2], 2:, :2] *= -1
    phase_matrix_expanded[:, :, :, inds[0], inds[1], inds[2], :2, 2:] *= -1

    expand_only = np.where(np.logical_or(np.logical_or(np.isclose(lon_scat, 0.0),
                                                       np.isclose(lon_scat, np.pi)),
                                         np.logical_or(np.isclose(scat_angles, 0.0),
                                                       np.isclose(scat_angles, np.pi))))

    phase_matrix_expanded[:, :, :, expand_only[0], expand_only[1], expand_only[2], :, :] = 0.0
    phase_matrix_expanded[:, :, :, expand_only[0], expand_only[1], expand_only[2], 0, 0] = f11[:, :, :, expand_only[0], expand_only[1], expand_only[2]]
    phase_matrix_expanded[:, :, :, expand_only[0], expand_only[1], expand_only[2], 0, 1] = f12[:, :, :, expand_only[0], expand_only[1], expand_only[2]]
    phase_matrix_expanded[:, :, :, expand_only[0], expand_only[1], expand_only[2], 1, 0] = f12[:, :, :, expand_only[0], expand_only[1], expand_only[2]]
    phase_matrix_expanded[:, :, :, expand_only[0], expand_only[1], expand_only[2], 1, 1] = f22[:, :, :, expand_only[0], expand_only[1], expand_only[2]]
    phase_matrix_expanded[:, :, :, expand_only[0], expand_only[1], expand_only[2], 2, 2] = f33[:, :, :, expand_only[0], expand_only[1], expand_only[2]]
    phase_matrix_expanded[:, :, :, expand_only[0], expand_only[1], expand_only[2], 2, 3] = f34[:, :, :, expand_only[0], expand_only[1], expand_only[2]]
    phase_matrix_expanded[:, :, :, expand_only[0], expand_only[1], expand_only[2], 3, 2] = -f34[:, :, :, expand_only[0], expand_only[1], expand_only[2]]
    phase_matrix_expanded[:, :, :, expand_only[0], expand_only[1], expand_only[2], 3, 3] = f44[:, :, :, expand_only[0], expand_only[1], expand_only[2]]

    return phase_matrix_expanded



