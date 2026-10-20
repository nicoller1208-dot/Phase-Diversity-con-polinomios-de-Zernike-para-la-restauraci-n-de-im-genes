"""Punto de entrada del proyecto de Phase Diversity."""
from __future__ import annotations

import argparse

from codigo.experimentos.comparacion import ejecutar


def parametros():
    parser = argparse.ArgumentParser(
        description=(
            "Phase Diversity con Zernike: comparación entre diversidad "
            "convencional y extendida."
        )
    )

    parser.add_argument("--imagen", type=str, default=None, help="Imagen propia opcional en PNG/JPG")
    parser.add_argument("--n", type=int, default=128, help="Resolución par >=64")
    parser.add_argument("--semilla", type=int, default=2026)
    parser.add_argument("--sigma", type=float, default=0.008, help="Desviación del ruido gaussiano en escala [0,1]")
    parser.add_argument("--lambda-ajuste", dest="lambda_ajuste", type=float, default=0.0003)
    parser.add_argument("--lambda-restauracion", dest="lambda_restauracion", type=float, default=0.04)
    parser.add_argument("--intentos", type=int, default=3)
    parser.add_argument("--iteraciones", type=int, default=100)
    parser.add_argument("--salida", type=str, default="resultados")

    args = parser.parse_args()

    if args.sigma <= 0 or args.lambda_ajuste <= 0 or args.lambda_restauracion <= 0:
        parser.error("sigma y ambas lambdas deben ser > 0")
    if args.intentos < 1 or args.iteraciones < 1:
        parser.error("intentos e iteraciones deben ser enteros positivos")

    return args


if __name__ == "__main__":
    ejecutar(parametros())
