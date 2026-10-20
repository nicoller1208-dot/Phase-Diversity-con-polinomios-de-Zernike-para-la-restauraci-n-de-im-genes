"""Parámetros físicos y numéricos del experimento de Phase Diversity."""
from __future__ import annotations

import numpy as np

# Modos ANSI/OSA estimados. Se excluyen pistón y tip/tilt.
MODOS = (3, 4, 5, 7, 8, 12)

NOMBRES = {
    3: "Astigmatismo oblicuo",
    4: "Desenfoque",
    5: "Astigmatismo horizontal/vertical",
    7: "Coma vertical",
    8: "Coma horizontal",
    12: "Aberración esférica primaria",
}

# Coeficientes verdaderos usados para generar los datos sintéticos [rad].
C_VERDADEROS = np.array([0.28, 0.40, -0.20, 0.24, -0.18, 0.14], dtype=float)

# Diversidades conocidas [rad].
D_DESENFOQUE = 1.30
D_ASTIGMATISMO = 0.85

# Radio normalizado de la pupila dentro de la malla numérica.
RADIO_PUPILA = 0.22
