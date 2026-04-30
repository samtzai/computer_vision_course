# -*- coding: utf-8 -*-
__author__ = "mi_nombre_aqui"

import matplotlib.pyplot as plt
import numpy as np
import skimage, skimage.io
from src.exercise_01.t01_load_image import visualizar_imagen
from pathlib import Path

file_img_01 = "./data/color/barrio_sesamo.jpg"
file_img_02 = "./data/color/espinete.jpg"

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)

def integrar_a_espinete():
    img_grupo = skimage.io.imread(file_img_01)
    img_espinete = skimage.io.imread(file_img_02)

    visualizar_imagen(img_grupo, titulo="imagen grupo")
    visualizar_imagen(img_espinete, titulo="imagen espinete")

    """
        NOMBRE:
        TODO:        
            Sobre la imagen de grupo
            Extraer los tres canales de color y pintarles por separado. 
            Guarda los tres canales de color por separado
    """

    # visualizar_imagen(canal_R, titulo='???', save_figure = True, figure_save_path = output_folder / 'fig_espinete_r.png')
    # visualizar_imagen(canal_G, titulo='???', save_figure = True, figure_save_path = output_folder / 'fig_espinete_g.png')
    # visualizar_imagen(canal_B, titulo='???', save_figure = True, figure_save_path = output_folder / 'fig_espinete_b.png')

    """
        NOMBRE:
        TODO:
            Sobre la imagen de grupo
            Pon a cero todos los canales menos uno y pinta la imagen resultante.
            Repite con las tres combinaciones posibles. 
    """

    # visualizar_imagen(canal_R, titulo='???', save_figure = True, figure_save_path = output_folder / 'fig_espinete_r2.png')
    # visualizar_imagen(canal_G, titulo='???', save_figure = True, figure_save_path = output_folder / 'fig_espinete_g2.png')
    # visualizar_imagen(canal_B, titulo='???', save_figure = True, figure_save_path = output_folder / 'fig_espinete_b2.png')

    """
        NOMBRE:
        TODO:
            Mediante slicing, extraer las caras de coco, epi, blas y triki y visualizar cada cara en un subplot.
            Graba las figuras resultantes
    """

    """
        NOMBRE:
        TODO:
            Espinete (al que no conocéis igual), no se encuentra en la foto de grupo, tratad de incluirle mediante slicing y visualizar la foto
            Bonus: Pega a Espinete de forma elegante mediante el uso de las máscaras que hemos usado en el otro script.
            Graba las figuras resultantes
            Explica el algoritmo desarrollado
            Respuesta:
    """


if __name__ == "__main__":
    integrar_a_espinete()
    # input()
