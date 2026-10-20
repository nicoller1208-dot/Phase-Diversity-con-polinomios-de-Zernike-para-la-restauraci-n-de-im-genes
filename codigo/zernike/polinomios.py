"""Polinomios de Zernike con indexación y normalización ANSI/OSA."""
from __future__ import annotations

from math import factorial
import numpy as np

from codigo.configuracion.parametros import MODOS, RADIO_PUPILA


def indice_osa(j: int) -> tuple[int, int]:
    """Convierte el índice secuencial ANSI/OSA j en el par (n,m)."""
    if j < 0:
        raise ValueError("El índice ANSI/OSA debe ser no negativo")

    for n in range(30):
        for m in range(-n, n + 1, 2):
            if (n * (n + 2) + m) // 2 == j:
                return n, m

    raise ValueError("Índice ANSI/OSA demasiado grande")


def zernike(
    j: int,
    rho: np.ndarray,
    theta: np.ndarray,
    pupila: np.ndarray,
) -> np.ndarray:
    """Calcula el modo real de Zernike Z_j sobre el disco unitario."""
    n, m = indice_osa(j)
    abs_m = abs(m)

    radial = np.zeros_like(rho, dtype=float)
    for s in range((n - abs_m) // 2 + 1):
        coef = (
            (-1) ** s
            * factorial(n - s)
            / (
                factorial(s)
                * factorial((n + abs_m) // 2 - s)
                * factorial((n - abs_m) // 2 - s)
            )
        )
        radial += coef * rho ** (n - 2 * s)

    # Normalización ANSI/OSA.
    norma = np.sqrt((n + 1) if m == 0 else 2 * (n + 1))
    angular = np.cos(m * theta) if m >= 0 else np.sin(abs_m * theta)

    return np.where(pupila, norma * radial * angular, 0.0)


def malla_pupila(n: int, radio: float = RADIO_PUPILA):
    """Construye la pupila circular y precalcula la base Zernike requerida."""
    coordenadas = np.arange(n) - n // 2
    yy, xx = np.meshgrid(coordenadas, coordenadas, indexing="ij")

    rho = np.hypot(xx, yy) / (radio * n)
    theta = np.arctan2(yy, xx)
    mascara = rho <= 1

    modos_requeridos = set(MODOS) | {4, 5}
    base = {j: zernike(j, rho, theta, mascara) for j in modos_requeridos}
    return mascara, base


def funcion_fase(c: np.ndarray, base: dict[int, np.ndarray]) -> np.ndarray:
    """Construye W(rho,theta)=sum_j C_j Z_j en radianes."""
    fase = np.zeros_like(base[4])
    for j, cj in zip(MODOS, c):
        fase += float(cj) * base[j]
    return fase
