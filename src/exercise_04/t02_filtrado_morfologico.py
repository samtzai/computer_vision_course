# -*- coding: utf-8 -*-
__author__ = 106360

from pathlib import Path

import cv2
import numpy as np
import skimage
import skimage.morphology
from matplotlib import pyplot as plt

from src.exercise_01.t01_load_image import visualizar_imagen, visualizar_imagenes

file_histo = "./data/histograma/cerebro_1.jpg"
file_thres = "./data/histograma/bookpage.jpg"
file_7seg = "./data/histograma/Segmentos7.jpg"

file_mariposa = "./data/morfologicos/mariposa.jpg"
file_mariposa_noisy = "./data/morfologicos/mariposa_noisy.jpg"

file_sudoku = "./data/morfologicos/sudoku.jpg"
file_windows = "./data/morfologicos/windows.jpg"

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)


def do_test_binary_morphology_filter_1():
    image_gray = skimage.color.rgb2gray(skimage.io.imread(file_sudoku))
    img_threshold = skimage.filters.threshold_local(image_gray, 65, offset=0)
    image_binary_adaptive = image_gray < img_threshold
    visualizar_imagenes([image_gray, image_binary_adaptive], ["orig", "binary"], 2, 1)

    """
    NOMBRE:
    TODO:
        Sobre la imagen binaria hacer lo siguiente:
        -  Aplicar un  filtro de erosión con tres tamaños de elemento estructural, incluyendo elementos rectangulares
        -  Aplicar un  filtro de dilatación con tres tamaños de elemento estructural, incluyendo elementos rectangulares
        -  Aplicar un  filtro de apertura con tres tamaños de elemento estructural, incluyendo elementos rectangulares
        -  Aplicar un  filtro de cierre con tres tamaños de elemento estructural, incluyendo elementos rectangulares
        - Guardar las imágenes en  en directorio out de la práctica como binary_morphology_filter_2_XXX.png    
        Respuesta:
    """


def do_test_gray_morphology():
    """
    NOMBRE:
    TODO:
        Una de las aplicaciones de las operaciones morfológicas enos permiten aislar el valor del fondo, es decir la iluminación no homogenea de la imagen.
        Esta operación consiste en restarle a una imagen una operación morfológica de ella misma con un elemento estructural grande.
        Revisa el código y prueba sus efectos con distintos tipos de elemento esturctural (distinto tamaño y distinta forma (p.e tamaño 1,9 y tamaño 9,1))
        Respuesta:
    """

    image_gray = skimage.color.rgb2gray(skimage.io.imread(file_sudoku))
    imagen_opening = skimage.morphology.dilation(image_gray, np.ones((9, 9)))
    imagen_removal = image_gray / (imagen_opening + 0.001)

    visualizar_imagenes(
        [image_gray, imagen_opening, imagen_removal], ["orig", "opening", "corregida"], 3, 1
    )


def do_test_binary_morphology_filter_2():
    image_gray = skimage.color.rgb2gray(skimage.io.imread(file_sudoku))
    img_threshold = skimage.filters.threshold_local(image_gray, 65, offset=0)
    image_binary_adaptive = image_gray < img_threshold
    visualizar_imagenes([image_gray, image_binary_adaptive], ["orig", "binary"], 2, 1)

    """
    NOMBRE:
    TODO:
        Sobre la imagen binaria diseñar la cadena de filtros más adecuada que permita quedarnos sólamente con los números del sudoku y la retícula
        Recodrad que podéis aplicar operaciones lógicas a los resultados de distintos filtros para combinarlos.
        - Explicar el proceso
        - Guardar las imágenes en el directorio out de la práctica como: binary_morphology_filter_2_XXX.png
        - Si queréis podéis también trabajar sobre la imagen en gris original antes de realizar el threshold. 
        ¿Podría ayudar el aplicar filtros morfológicos en niveles de gris antes de aplicar el threshold?
        Respuesta:

    """


if __name__ == "__main__":
    do_test_gray_morphology()
    # do_test_binary_morphology_filter_1()
