"""Ejecuta el experimento principal desde la raíz del repositorio."""
from codigo.main import parametros
from codigo.experimentos.comparacion import ejecutar


if __name__ == "__main__":
    ejecutar(parametros())
