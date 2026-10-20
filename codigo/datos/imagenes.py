"""Carga y preprocesamiento del objeto ideal."""
from __future__ import annotations

import numpy as np
from skimage import color, data, io, transform


def objeto_original(n: int, archivo: str | None = None) -> np.ndarray:
    """Carga una imagen propia o usa skimage.data.camera por defecto."""
    if archivo:
        imagen = io.imread(archivo)
        if imagen.ndim == 3:
            imagen = color.rgb2gray(imagen[:, :, :3])

        imagen = imagen.astype(float)
        if imagen.max() > 1:
            imagen /= imagen.max()
    else:
        imagen = data.camera().astype(float) / 255.0

    return transform.resize(
        imagen,
        (n, n),
        anti_aliasing=True,
        preserve_range=True,
    ).clip(0, 1)
