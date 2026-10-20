"""Generación de figuras y archivos de resultados."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from codigo.configuracion.parametros import (
    C_VERDADEROS,
    D_ASTIGMATISMO,
    D_DESENFOQUE,
    MODOS,
    NOMBRES,
)
from codigo.zernike.polinomios import funcion_fase, indice_osa


def guardar_figuras_y_tablas(
    salida: Path,
    objeto: np.ndarray,
    mascara: np.ndarray,
    base: dict[int, np.ndarray],
    esquemas: dict[str, list[np.ndarray]],
    resultados: dict,
    args,
) -> None:
    """Guarda figuras en resultados/figuras y datos numéricos en resultados/tablas."""
    carpeta_figuras = salida / "figuras"
    carpeta_tablas = salida / "tablas"
    carpeta_figuras.mkdir(parents=True, exist_ok=True)
    carpeta_tablas.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for fila, nombre in enumerate(esquemas):
        r = resultados[nombre]
        mapas = [objeto, r["datos"][0], r["datos"][1], r["restaurada"]]
        titulos = [
            "Objeto original",
            "Adquisición sin diversidad",
            "Adquisición con diversidad",
            "Imagen restaurada",
        ]
        for ax, imagen, titulo in zip(axes[fila], mapas, titulos):
            ax.imshow(imagen, cmap="gray", vmin=0, vmax=1)
            ax.set_title(titulo, fontsize=10)
            ax.axis("off")

        axes[fila, 0].set_ylabel(nombre, fontsize=12)
        axes[fila, 3].set_title(
            f"Restaurada: PSNR {r['restauracion']['PSNR_dB']:.2f} dB\n"
            f"SSIM {r['restauracion']['SSIM']:.3f}",
            fontsize=10,
        )

    fig.suptitle(
        "Phase Diversity: comparación con dos adquisiciones por método",
        fontsize=14,
    )
    fig.savefig(carpeta_figuras / "comparacion_imagenes.png", dpi=165)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), constrained_layout=True)
    w_true = np.where(mascara, funcion_fase(C_VERDADEROS, base), np.nan)
    maxabs = max(1e-12, float(np.nanmax(np.abs(w_true))))

    im = axes[0].imshow(w_true, cmap="RdBu_r", vmin=-maxabs, vmax=maxabs)
    axes[0].set_title("Frente de onda verdadero (rad)")
    fig.colorbar(im, ax=axes[0], shrink=0.72)

    for i, nombre in enumerate(esquemas, 1):
        estimados = np.array(resultados[nombre]["coeficientes_estimados_rad"])
        w_est = np.where(mascara, funcion_fase(estimados, base), np.nan)
        im = axes[i].imshow(w_est, cmap="RdBu_r", vmin=-maxabs, vmax=maxabs)
        axes[i].set_title(f"Estimación {nombre.lower()} (rad)")
        fig.colorbar(im, ax=axes[i], shrink=0.72)

    eje = np.arange(len(MODOS))
    axes[3].plot(eje, C_VERDADEROS, "o-", label="Real")
    for nombre in esquemas:
        axes[3].plot(
            eje,
            resultados[nombre]["coeficientes_estimados_rad"],
            "o--",
            label=nombre,
        )
    axes[3].set_xticks(eje, [f"Z{j}" for j in MODOS])
    axes[3].set_ylabel("Coeficientes (rad)")
    axes[3].set_title("Recuperación de aberraciones")
    axes[3].grid(alpha=0.25)
    axes[3].legend(fontsize=8)

    for ax in axes[:3]:
        ax.axis("off")

    fig.savefig(carpeta_figuras / "estimacion_zernike.png", dpi=165)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.1), constrained_layout=True)
    for fila, (nombre, diversidades) in enumerate(esquemas.items()):
        r = resultados[nombre]
        fase_segunda = np.where(
            mascara,
            funcion_fase(C_VERDADEROS, base) + diversidades[1],
            np.nan,
        )
        psf_0, psf_1 = r["psf"]
        otf_1 = np.fft.fft2(np.fft.ifftshift(psf_1))

        escenas = [
            fase_segunda,
            np.log10(psf_0 + 1e-12),
            np.log10(psf_1 + 1e-12),
            np.abs(np.fft.fftshift(otf_1)),
        ]
        titulos = [
            "Fase total segunda pupila (rad)",
            "log10 PSF referencia",
            "log10 PSF diversidad",
            "|OTF| segunda adquisición",
        ]

        for ax, array, titulo in zip(axes[fila], escenas, titulos):
            ax.imshow(array, cmap="viridis")
            ax.set_title(titulo, fontsize=10)
            ax.axis("off")
        axes[fila, 0].set_ylabel(nombre, fontsize=12)

    fig.suptitle("Diagnóstico del modelo de pupila, PSF y OTF", fontsize=13)
    fig.savefig(carpeta_figuras / "diagnostico_optico.png", dpi=160)
    plt.close(fig)

    resumen = {
        nombre: {
            clave: valor
            for clave, valor in r.items()
            if clave not in ("datos", "restaurada", "psf")
        }
        for nombre, r in resultados.items()
    }
    resumen["configuracion"] = {
        "n": args.n,
        "semilla": args.semilla,
        "sigma": args.sigma,
        "modos_ANSI_OSA": list(MODOS),
        "diversidad_Z4_rad": D_DESENFOQUE,
        "diversidad_extra_Z5_rad": D_ASTIGMATISMO,
        "lambda_ajuste": args.lambda_ajuste,
        "lambda_restauracion": args.lambda_restauracion,
        "modelo_bordes": "convolución circular",
        "K": 2,
        "imagen": args.imagen if args.imagen else "skimage.data.camera()",
    }

    (carpeta_tablas / "resultados.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2),
        encoding="utf8",
    )

    with (carpeta_tablas / "coeficientes_zernike.csv").open(
        "w",
        encoding="utf8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "j_ANSI_OSA",
            "n",
            "m",
            "nombre",
            "valor_real_rad",
            "convencional_rad",
            "extendido_rad",
        ])
        for i, j in enumerate(MODOS):
            n_z, m_z = indice_osa(j)
            writer.writerow([
                j,
                n_z,
                m_z,
                NOMBRES[j],
                float(C_VERDADEROS[i]),
                resultados["Convencional"]["coeficientes_estimados_rad"][i],
                resultados["Extendido"]["coeficientes_estimados_rad"][i],
            ])
