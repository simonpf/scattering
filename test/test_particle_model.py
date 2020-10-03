"""
Test for ParticleModel class defined in particle_model.h.
"""
import os
import numpy as np

from utils import RANDOM_DATA_PATH, AZIMUTHALLY_RANDOM_DATA_PATH
from scatlib.arts_ssdb import HabitFolder, ParticleFile

def test_random_data():
    """
    Load test data as scattering model and ensure that the parsed meta
    data as well as the scattering data is the same as when the data
    is loaded directly.
    """
    habit_folder = HabitFolder(RANDOM_DATA_PATH)
    particle_model = habit_folder.to_particle_model()
    assert all(particle_model.get_d_eq() == [101 * 1e-6, 770 * 1e-6])
    assert all(particle_model.get_d_max() == [191 * 1e-6, 1014 * 1e-6])
    assert all(particle_model.get_mass() == [5.0044e-10, 2.19345e-7])

    path = os.path.join(RANDOM_DATA_PATH,
                        "Dmax00191um_Dveq00101um_Mass5.00440e-10kg.nc")
    p = particle_model.get_single_scattering_data(0)
    p_ref = ParticleFile(path).to_single_scattering_data()

    assert np.all(np.isclose(p.get_phase_matrix(),
                             p_ref.get_phase_matrix()))

    path = os.path.join(RANDOM_DATA_PATH,
                        "Dmax01014um_Dveq00770um_Mass2.19345e-07kg.nc")
    p = particle_model.get_single_scattering_data(1)
    p_ref = ParticleFile(path).to_single_scattering_data()

    assert np.all(np.isclose(p.get_phase_matrix(),
                             p_ref.get_phase_matrix()))

def test_azimuthally_random_data():
    """
    Load test data as scattering model and ensure that the parsed meta
    data as well as the scattering data is the same as when the data
    is loaded directly.
    """
    habit_folder = HabitFolder(AZIMUTHALLY_RANDOM_DATA_PATH)
    particle_model = habit_folder.to_particle_model()
    assert all(particle_model.get_d_eq() == [101 * 1e-6, 770 * 1e-6])
    assert all(particle_model.get_d_max() == [191 * 1e-6, 1014 * 1e-6])
    assert all(particle_model.get_mass() == [5.00440e-10, 2.19345e-7])

    path = os.path.join(AZIMUTHALLY_RANDOM_DATA_PATH,
                        "Dmax00191um_Dveq00101um_Mass5.00440e-10kg.nc")
    p = particle_model.get_single_scattering_data(0)
    p_ref = ParticleFile(path).to_single_scattering_data()

    assert np.all(np.isclose(p.get_phase_matrix(),
                             p_ref.get_phase_matrix()))

    path = os.path.join(AZIMUTHALLY_RANDOM_DATA_PATH,
                        "Dmax01014um_Dveq00770um_Mass2.19345e-07kg.nc")
    p = particle_model.get_single_scattering_data(1)
    p_ref = ParticleFile(path).to_single_scattering_data()

    assert np.all(np.isclose(p.get_phase_matrix(),
                             p_ref.get_phase_matrix()))
