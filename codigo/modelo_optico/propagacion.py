"""Modelo óptico directo: pupila compleja, PSF, OTF y adquisiciones."""
from __future__ import annotations

import numpy as np

from codigo.zernike.polinomios import funcion_fase


def construir_otf(fase: np.ndarray, mascara: np.ndarray):
    """Calcula PSF incoherente y OTF a partir de una fase de pupila.

    P = A exp(i fase)
    h = IFFT{P}
    PSF = |h|^2
    OTF = FFT{PSF}
    """
    pupila_compleja = mascara * np.exp(1j * fase)

    h = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(pupila_compleja))
    )

    psf = np.abs(h) ** 2
    psf /= psf.sum()
    otf = np.fft.fft2(np.fft.ifftshift(psf))
    return psf, otf


def generar_otfs(
    c: np.ndarray,
    diversidades: list[np.ndarray],
    mascara: np.ndarray,
    base: dict[int, np.ndarray],
):
    """Genera PSF y OTF para todos los canales de Phase Diversity."""
    w = funcion_fase(c, base)
    psfs, otfs = [], []

    for fase_conocida in diversidades:
        psf, otf = construir_otf(w + fase_conocida, mascara)
        psfs.append(psf)
        otfs.append(otf)

    return np.stack(psfs), np.stack(otfs)


def simular_datos(
    objeto: np.ndarray,
    otfs: np.ndarray,
    ruido_base: np.ndarray,
):
    """Simula d_k=f*PSF_k+n_k mediante multiplicación en Fourier."""
    f_fft = np.fft.fft2(objeto)
    limpio = np.stack([
        np.fft.ifft2(f_fft * otf).real
        for otf in otfs
    ])
    return limpio + ruido_base, limpio
