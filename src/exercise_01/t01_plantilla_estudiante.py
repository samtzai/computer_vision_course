# -*- coding: utf-8 -*-
__author__ = "mi_nombre_aqui"

"""
Date = 17/07/2018
Author = xxxxxxx
Project = Práctica 01, principios de python
"""
from pathlib import Path

from .t01_load_image import cargar_imagen, visualizar_imagen

if __name__ == "__main__":
    """
       TODO: Modifica el script para que cargue una imagen cualquiera y guarda la figura que se 
       visualiza incluyendo en el título el nombre de estudiante
        	Escribir por pantalla el valor del píxel fila=30, columna=50 de la imagen.
        	Pintar la imagen con el nombre del estudiante y el valor del pixel en el título y almacenar la figura generada en el directorio de salida
        	Sube el código y las imágenes generadas a tu repositorio de código.
    """
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    file_path = "./data/underwater/Ancuti01.png"
    imagen_rgb = cargar_imagen(file_path)
    visualizar_imagen(
        imagen_rgb,
        titulo="Artzai",
        save_figure=True,
        figure_save_path=output_folder / "t02_image_out_exercise_02.png",
    )
