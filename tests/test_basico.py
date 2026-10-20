import numpy as np

from codigo.zernike.polinomios import indice_osa, malla_pupila
from codigo.modelo_optico.propagacion import construir_otf


def test_indices_ansi_osa():
    assert indice_osa(4) == (2, 0)
    assert indice_osa(5) == (2, 2)
    assert indice_osa(7) == (3, -1)
    assert indice_osa(8) == (3, 1)


def test_psf_normalizada():
    mascara, base = malla_pupila(64)
    psf, otf = construir_otf(np.zeros_like(base[4]), mascara)
    assert np.isclose(psf.sum(), 1.0)
    assert np.isclose(np.abs(otf[0, 0]), 1.0)
