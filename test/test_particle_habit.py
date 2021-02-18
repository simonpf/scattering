"""
Test for ParticleModel class defined in particle_model.h.
"""
import os
import numpy as np

from utils import RANDOM_DATA_PATH, AZIMUTHALLY_RANDOM_DATA_PATH
from scattering.arts_ssdb import HabitFolder, ParticleFile


class TestRandomData():
    """
    Tests for totally random scattering data.
    """
    def setup_method(self):
        #
        # Particle model
        #
        self.habit_folder = HabitFolder(RANDOM_DATA_PATH)
        self.particle_model = self.habit_folder.to_particle_habit()

        #
        # First particle
        #
        path = os.path.join(RANDOM_DATA_PATH,
                            "Dmax00688um_Dveq00361um_Mass2.25360e-08kg.nc")
        self.particle_1 = self.particle_model.get_single_scattering_data(0)
        self.particle_1_ref = ParticleFile(path).to_single_scattering_data()

        #
        # Second particle
        #
        path = os.path.join(RANDOM_DATA_PATH,
                            "Dmax03369um_Dveq00771um_Mass2.19881e-07kg.nc")
        self.particle_2 = self.particle_model.get_single_scattering_data(1)
        self.particle_2_ref = ParticleFile(path).to_single_scattering_data()

    def test_load(self):
        """
        Load test data as scattering model and ensure that the parsed meta
        data as well as the scattering data is the same as when the data
        is loaded directly.
        """
        assert all(self.particle_model.get_d_eq() == [361 * 1e-6, 771 * 1e-6])
        assert all(self.particle_model.get_d_max() == [688 * 1e-6, 3369 * 1e-6])
        assert all(self.particle_model.get_mass() == [2.2536e-8, 2.19881e-7])

        assert np.all(np.isclose(self.particle_1.get_phase_matrix_data(),
                                 self.particle_1_ref.get_phase_matrix_data()))
        assert np.all(np.isclose(self.particle_2.get_phase_matrix_data(),
                                 self.particle_2_ref.get_phase_matrix_data()))

    def test_calculate_bulk_properties(self):
        """
        Tests calculation of bulk properties ensuring that the phase matrices are
        added according to provided particle densities.
        """
        habit_folder = HabitFolder(RANDOM_DATA_PATH)
        particle_model = habit_folder.to_particle_habit()
        assert all(self.particle_model.get_d_eq() == [361 * 1e-6, 771 * 1e-6])
        assert all(self.particle_model.get_d_max() == [688 * 1e-6, 3369 * 1e-6])
        assert all(self.particle_model.get_mass() == [2.2536e-8, 2.19881e-7])

        props_1 = self.particle_model.calculate_bulk_properties(230, [1e3, 1e3])
        phase_matrix = props_1.get_phase_matrix_data()

        particle = self.particle_1.copy()
        particle += self.particle_2
        particle.normalize(1.0)
        phase_matrix_ref = particle.get_phase_matrix_data()[:, [0], ...]

        assert np.all(np.isclose(phase_matrix, phase_matrix_ref))

    def test_extract_scattering_coeffs(self):
        """
        Tests extraction of scattering coefficients from the scattering
        field.
        """
        particle_model = self.particle_model.set_stokes_dim(1)
        for i in range(2):
            sd = particle_model.get_single_scattering_data(i)
            pm = sd.get_phase_matrix_data()[..., 0]
            sd_ref = self.particle_model.get_single_scattering_data(i)
            pm_ref = sd_ref.get_phase_matrix_data()[..., 0]
            assert np.all(np.isclose(pm, pm_ref))



class TestAzimuthallyRandomData():
    """
    Test for azimuthally random scattering data.
    """
    def setup_method(self):
        #
        # Particle model
        #
        self.habit_folder = HabitFolder(AZIMUTHALLY_RANDOM_DATA_PATH)
        self.particle_model = self.habit_folder.to_particle_habit()

        #
        # First particle
        #
        path = os.path.join(AZIMUTHALLY_RANDOM_DATA_PATH,
                            "Dmax00590um_Dveq00251um_Mass7.59425e-09kg.nc")
        self.particle_1 = self.particle_model.get_single_scattering_data(0)
        self.particle_1_ref = ParticleFile(path).to_single_scattering_data()

        #
        # Second particle
        #
        path = os.path.join(AZIMUTHALLY_RANDOM_DATA_PATH,
                            "Dmax04151um_Dveq01257um_Mass9.53208e-07kg.nc")
        self.particle_2 = self.particle_model.get_single_scattering_data(1)
        self.particle_2_ref = ParticleFile(path).to_single_scattering_data()

    def test_load(self):
        """
        Load test data as scattering model and ensure that the parsed meta
        data as well as the scattering data is the same as when the data
        is loaded directly.
        """
        assert all(self.particle_model.get_d_eq() == [251 * 1e-6, 1257 * 1e-6])
        assert all(self.particle_model.get_d_max() == [590 * 1e-6, 4151 * 1e-6])
        assert all(self.particle_model.get_mass() == [7.59425e-9, 9.53208e-7])

        assert np.all(np.isclose(self.particle_1.get_phase_matrix_data(),
                                 self.particle_1_ref.get_phase_matrix_data()))
        assert np.all(np.isclose(self.particle_2.get_phase_matrix_data(),
                                 self.particle_2_ref.get_phase_matrix_data()))

    def test_calculate_bulk_properties(self):
        """
        Tests calculation of bulk properties ensuring that the phase matrices are
        added according to provided particle densities.
        """
        habit_folder = HabitFolder(AZIMUTHALLY_RANDOM_DATA_PATH)
        particle_model = habit_folder.to_particle_habit()
        assert all(self.particle_model.get_d_eq() == [251 * 1e-6, 1257 * 1e-6])
        assert all(self.particle_model.get_d_max() == [590 * 1e-6, 4151 * 1e-6])
        assert all(self.particle_model.get_mass() == [7.59425e-9, 9.53208e-7])

        props_1 = self.particle_model.calculate_bulk_properties(230, [1e3, 1e3])
        phase_matrix = props_1.get_phase_matrix_data()

        particle = self.particle_1.copy()
        particle += self.particle_2
        particle.normalize(1.0)
        phase_matrix_ref = particle.get_phase_matrix_data()[:, [0], ...]

        return phase_matrix, phase_matrix_ref

    def test_extract_scattering_coeffs(self):
        """
        Tests extraction of scattering coefficients from the scattering
        field.
        """
        particle_model = self.particle_model.set_stokes_dim(1)
        for i in range(2):
            sd = particle_model.get_single_scattering_data(i)
            pm = sd.get_phase_matrix_data()[..., 0]
            sd_ref = self.particle_model.get_single_scattering_data(i)
            pm_ref = sd_ref.get_phase_matrix_data()[..., 0]
            assert np.all(np.isclose(pm, pm_ref))
