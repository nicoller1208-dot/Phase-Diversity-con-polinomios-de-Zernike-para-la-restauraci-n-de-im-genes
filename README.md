# Phase Diversity con polinomios de Zernike para la restauración de imágenes

**Autores:** David Restrepo C. y Nicolle Rodríguez E.  
**Universidad EAFIT — Ingeniería Física**

Este repositorio contiene la implementación numérica del modelo de Phase Diversity para un sistema óptico incoherente. Se comparan dos esquemas de diversidad: el convencional con desenfoque conocido y una variante extendida con desenfoque más astigmatismo conocido.

## Organización del repositorio

```text
.
├── codigo/
│   ├── configuracion/       # parámetros físicos y numéricos
│   ├── zernike/             # ANSI/OSA y construcción de W(rho,theta)
│   ├── modelo_optico/       # pupila, PSF, OTF y modelo directo
│   ├── datos/               # carga y preprocesamiento de imágenes
│   ├── phase_diversity/     # problema inverso y optimización
│   ├── evaluacion/          # métricas, figuras y archivos de salida
│   ├── experimentos/        # comparación convencional vs. extendido
│   └── main.py              # interfaz por línea de comandos
├── resultados/
│   ├── figuras/             # imágenes generadas por el código
│   └── tablas/              # CSV y JSON con resultados
├── documentacion/
│   ├── informe/             # espacio para el informe académico
│   └── referencias/         # espacio para bibliografía y artículos
├── tests/                   # pruebas básicas del modelo
├── run.py                   # ejecución desde la raíz
├── requirements.txt
└── README.md
```

## Ejecución

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el experimento completo:

```bash
python run.py
```

También puede usarse una imagen propia:

```bash
python run.py --imagen ruta/a/imagen.png
```

Los resultados se guardan automáticamente en `resultados/figuras/` y `resultados/tablas/`.

## Flujo matemático

La aberración se representa como

\[
W(\rho,\theta)=\sum_j C_j Z_j(\rho,\theta),
\]

y para cada canal de diversidad se construye

\[
P_k=A\exp\{i[W+\theta_k]\}.
\]

A partir de la pupila se calculan la respuesta coherente, la PSF y la OTF, y finalmente se generan las adquisiciones degradadas. El problema inverso estima los coeficientes de Zernike usando L-BFGS-B y reconstruye el objeto en el dominio de Fourier.

## Estado actual

La implementación actual corresponde a una validación numérica sintética. La comparación sistemática entre diferentes magnitudes y tipos de diversidad, así como la validación experimental, quedan como trabajo posterior.
