"""Métricas cuantitativas de validación."""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def puntajes(objeto: np.ndarray, imagen: np.ndarray) -> dict[str, float]:
    """Calcula PSNR, SSIM y RMSE de imagen."""
    return {
        "PSNR_dB": float(
            peak_signal_noise_ratio(objeto, imagen, data_range=1.0)
        ),
        "SSIM": float(
            structural_similarity(objeto, imagen, data_range=1.0)
        ),
        "RMSE_imagen": float(
            np.sqrt(np.mean((objeto - imagen) ** 2))
        ),
    }
