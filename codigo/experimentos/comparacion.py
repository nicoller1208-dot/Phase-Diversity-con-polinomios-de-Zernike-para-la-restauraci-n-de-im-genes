"""Orquestación del experimento convencional vs. extendido."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from codigo.configuracion.parametros import C_VERDADEROS, D_ASTIGMATISMO, D_DESENFOQUE, MODOS, NOMBRES
from codigo.datos.imagenes import objeto_original
from codigo.evaluacion.metricas import puntajes
from codigo.evaluacion.graficas import guardar_figuras_y_tablas
from codigo.phase_diversity.estimacion import ajustar_aberraciones, restaurar
from codigo.modelo_optico.propagacion import generar_otfs, simular_datos
from codigo.zernike.polinomios import indice_osa, malla_pupila


def ejecutar(args):
    """Ejecuta simulación, estimación, restauración y validación."""
    n = args.n
    if n < 64 or n % 2 != 0:
        raise ValueError("n debe ser par y >= 64")

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    mascara, base = malla_pupila(n)
    objeto = objeto_original(n, args.imagen)

    cero = np.zeros((n, n))
    esquemas = {
        "Convencional": [
            cero,
            D_DESENFOQUE * base[4],
        ],
        "Extendido": [
            cero,
            D_DESENFOQUE * base[4] + D_ASTIGMATISMO * base[5],
        ],
    }

    ruido_base = np.random.default_rng(args.semilla).normal(
        0,
        args.sigma,
        (2, n, n),
    )

    resultados = {}

    print("\nÍndice ANSI/OSA de los modos estimados:")
    for j, cj in zip(MODOS, C_VERDADEROS):
        print(
            f"  j={j:2d}  (n,m)={indice_osa(j)!s:>8}  "
            f"C_real={cj:+.3f} rad | {NOMBRES[j]}"
        )

    for nombre, diversidades in esquemas.items():
        psfs_true, otfs_true = generar_otfs(
            C_VERDADEROS,
            diversidades,
            mascara,
            base,
        )

        datos, _ = simular_datos(objeto, otfs_true, ruido_base)

        opt, costos = ajustar_aberraciones(
            datos,
            diversidades,
            mascara,
            base,
            args.sigma,
            args.lambda_ajuste,
            args.semilla + 159,
            args.intentos,
            args.iteraciones,
        )

        estimados = np.asarray(opt.x)
        restauracion = restaurar(
            datos,
            estimados,
            diversidades,
            mascara,
            base,
            args.lambda_restauracion,
        )

        rmse_c = float(np.sqrt(np.mean((estimados - C_VERDADEROS) ** 2)))

        resultados[nombre] = {
            "coeficientes_reales_rad": C_VERDADEROS.tolist(),
            "coeficientes_estimados_rad": estimados.tolist(),
            "RMSE_coeficientes_rad": rmse_c,
            "optimizador_convergio": bool(opt.success),
            "mensaje_optimizador": str(opt.message),
            "iteraciones": int(opt.nit),
            "costos_inicializaciones": costos,
            "adquisicion_0": puntajes(objeto, datos[0].clip(0, 1)),
            "restauracion": puntajes(objeto, restauracion),
            "diversidades": (
                "[0, 1.30 Z4]"
                if nombre == "Convencional"
                else "[0, 1.30 Z4 + 0.85 Z5]"
            ),
            "datos": datos,
            "restaurada": restauracion,
            "psf": psfs_true,
        }

        print(
            f"\n{nombre}: éxito={opt.success}, iter={opt.nit}, "
            f"RMSE_C={rmse_c:.5f} rad"
        )
        print("  real    :", np.round(C_VERDADEROS, 4))
        print("  estimada:", np.round(estimados, 4))
        print(
            "  PSNR sin restaurar: ",
            f"{resultados[nombre]['adquisicion_0']['PSNR_dB']:.2f} dB",
        )
        print(
            "  PSNR restaurada:    ",
            f"{resultados[nombre]['restauracion']['PSNR_dB']:.2f} dB",
        )
        print(
            "  SSIM restaurada:    ",
            f"{resultados[nombre]['restauracion']['SSIM']:.4f}",
        )

    guardar_figuras_y_tablas(
        salida,
        objeto,
        mascara,
        base,
        esquemas,
        resultados,
        args,
    )

    print(f"\nArchivos guardados en: {salida.resolve()}")
    print(
        "ADVERTENCIA: esta es una validación numérica sintética; "
        "todavía no es validación experimental."
    )

    return resultados
