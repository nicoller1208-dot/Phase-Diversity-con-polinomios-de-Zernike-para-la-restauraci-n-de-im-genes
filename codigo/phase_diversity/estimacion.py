"""Modelo inverso: estimación perfilada del objeto y de aberraciones."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from codigo.configuracion.parametros import MODOS
from codigo.modelo_optico.propagacion import generar_otfs


def estimar_objeto_fft(
    datos_fft: np.ndarray,
    otfs: np.ndarray,
    lambda_tikhonov: float,
) -> np.ndarray:
    """Estima el objeto en Fourier para un conjunto fijo de OTF."""
    numerador = np.sum(np.conj(otfs) * datos_fft, axis=0)
    denominador = np.sum(np.abs(otfs) ** 2, axis=0) + lambda_tikhonov
    return numerador / denominador


def ajustar_aberraciones(
    datos: np.ndarray,
    diversidades: list[np.ndarray],
    mascara: np.ndarray,
    base: dict[int, np.ndarray],
    sigma: float,
    lambda_ajuste: float,
    semilla: int,
    intentos: int = 3,
    iteraciones: int = 100,
):
    """Estima los coeficientes Zernike mediante L-BFGS-B."""
    d_fft = np.fft.fft2(datos, axes=(-2, -1))
    potencia = np.sum(np.abs(d_fft) ** 2, axis=0)

    n = datos.shape[-1]
    escalado = n * n * len(diversidades) * sigma**2

    def objetivo(c: np.ndarray) -> float:
        _, otfs = generar_otfs(c, diversidades, mascara, base)

        numerador = np.sum(np.conj(otfs) * d_fft, axis=0)
        denominador = np.sum(np.abs(otfs) ** 2, axis=0) + lambda_ajuste
        termino = potencia - np.abs(numerador) ** 2 / denominador
        return float(np.mean(termino) / escalado)

    rng = np.random.default_rng(semilla)
    inicios = [np.zeros(len(MODOS))]
    for _ in range(intentos - 1):
        inicios.append(rng.uniform(-0.42, 0.42, size=len(MODOS)))

    candidatos = []
    for inicio in inicios:
        solucion = minimize(
            objetivo,
            inicio,
            method="L-BFGS-B",
            bounds=[(-0.85, 0.85)] * len(MODOS),
            options={
                "maxiter": iteraciones,
                "ftol": 1e-11,
                "gtol": 1e-6,
                "maxls": 25,
            },
        )
        candidatos.append(solucion)

    mejor = min(candidatos, key=lambda s: s.fun)
    return mejor, [float(s.fun) for s in candidatos]


def restaurar(
    datos: np.ndarray,
    coeficientes: np.ndarray,
    diversidades: list[np.ndarray],
    mascara: np.ndarray,
    base: dict[int, np.ndarray],
    lambda_restauracion: float,
) -> np.ndarray:
    """Reconstruye la imagen usando los coeficientes estimados."""
    _, otfs = generar_otfs(coeficientes, diversidades, mascara, base)

    f_fft = estimar_objeto_fft(
        np.fft.fft2(datos, axes=(-2, -1)),
        otfs,
        lambda_restauracion,
    )

    return np.fft.ifft2(f_fft).real.clip(0, 1)
