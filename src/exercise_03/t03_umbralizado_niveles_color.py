# -----------------------
#    Date = 17/07/2018
#  Author = 106376
# Project = code
# -----------------------

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
from skimage.filters import threshold_local, threshold_otsu

from src.exercise_01.t01_load_image import visualizar_imagen, visualizar_imagenes
from pathlib import Path
output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)

file_bodegon = "./data/color/bodegon.png"
file_manzanas = "./data/color/Manzanas.jpg"


def do_test09():
    """
    NOMBRE:
    TODO:
        Comentar qué hace el código.
        Guardar la mejor imagen resultante en: /out/practica03/fig_test09_nombrexx.png
        Respuesta:
    """

    img_in = skimage.io.imread(file_manzanas)
    visualizar_imagen(img_in, "manzanas")
    canal_R = img_in[:, :, 0]
    canal_G = img_in[:, :, 1]
    canal_B = img_in[:, :, 2]
    visualizar_imagenes(
        [img_in, None, None, canal_R, canal_G, canal_B],
        ["Orig", "None", "None", "R", "G", "B"],
        3,
        2,
    )

    img_in = skimage.io.imread(file_manzanas)
    visualizar_imagen(img_in, "manzanas")
    imagen_R = img_in.copy()
    imagen_G = img_in.copy()
    imagen_B = img_in.copy()

    imagen_R[:, :, 1] = 0
    imagen_R[:, :, 2] = 0
    imagen_G[:, :, 0] = 0
    imagen_G[:, :, 2] = 0
    imagen_B[:, :, 0] = 0
    imagen_B[:, :, 1] = 0
    visualizar_imagenes(
        [img_in, None, None, imagen_R, imagen_G, imagen_B],
        ["Orig", "None", "None", "R", "G", "B"],
        3,
        2,
    )


def do_test10():
    """
    NOMBRE:
    TODO:
        ¿Qué ocurre en la imagen de tono en la manzana roja, por qué?
        Guardar la mejor imagen resultante en: /out/practica03/fig_test10_nombrexx.png
        Respuesta:
    """

    img_in = skimage.io.imread(file_manzanas)
    visualizar_imagen(img_in, "manzanas")
    canal_R = img_in[:, :, 0]
    canal_G = img_in[:, :, 1]
    canal_B = img_in[:, :, 2]
    visualizar_imagenes(
        [img_in, None, None, canal_R, canal_G, canal_B],
        ["Orig", "None", "None", "R", "G", "B"],
        3,
        2,
    )

    img_HSV = skimage.color.rgb2hsv(img_in)
    canal_H = img_HSV[:, :, 0]
    canal_S = img_HSV[:, :, 1]
    canal_V = img_HSV[:, :, 2]
    visualizar_imagenes(
        [img_in, img_HSV, None, canal_H, canal_S, canal_V],
        ["Orig", "HSV", "None", "H", "S", "V"],
        3,
        2,
    )


def do_test11():
    """
    NOMBRE:
    TODO:
        Explica lo que aparece en los canales L a y b y por qué
        Guardar la mejor imagen resultante en: /out/practica03/fig_test11_nombrexx.png
        Respuesta:
    """

    img_in = skimage.io.imread(file_manzanas)
    visualizar_imagen(img_in, "manzanas")
    canal_R = img_in[:, :, 0]
    canal_G = img_in[:, :, 1]
    canal_B = img_in[:, :, 2]
    visualizar_imagenes(
        [img_in, None, None, canal_R, canal_G, canal_B],
        ["Orig", "None", "None", "R", "G", "B"],
        3,
        2,
    )

    img_Lab = skimage.color.rgb2lab(img_in)
    canal_L = img_Lab[:, :, 0]
    canal_a = img_Lab[:, :, 1]
    canal_b = img_Lab[:, :, 2]
    visualizar_imagenes(
        [img_in, img_Lab, None, canal_L, canal_a, canal_b],
        ["Orig", "Lab", "None", "L", "a", "b"],
        3,
        2,
    )


def do_test12():
    """
    NOMBRE:
    TODO:
        Ajustar los valores de threshold para segmentar la manzana roja.
        Prestad atención al uso de operaciones lógicas en la segmentación con varios thresholds.
        Guardar la mejor imagen resultante en: /out/practica03/fig_test12_nombrexx.png
        Respuesta:
    """

    img_in = skimage.io.imread(file_manzanas)

    img_Lab = skimage.color.rgb2lab(img_in)
    canal_L = img_Lab[:, :, 0]
    canal_a = img_Lab[:, :, 1]
    canal_b = img_Lab[:, :, 2]

    th_L_max = 100
    th_L_min = 0
    th_a_max = 0
    th_a_min = -70
    th_b_max = 70
    th_b_min = -70

    seg_L = np.logical_and(canal_L < th_L_max, canal_L > th_L_min)
    seg_a = np.logical_and(canal_a < th_a_max, canal_a > th_a_min)
    seg_b = np.logical_and(canal_b < th_b_max, canal_b > th_b_min)
    imagen_segmentada = np.logical_and(np.logical_and(seg_a, seg_b), seg_L)
    visualizar_imagenes(
        [img_in, img_Lab, imagen_segmentada, canal_L, canal_a, canal_b],
        ["Orig", "Lab", "Segmentada", "L", "a", "b"],
        3,
        2,
    )


def do_test13():
    """
    NOMBRE:
    TODO:
        Seleccionando la imagen bodegon, selecciona una fruta y segmentala en al menos dos modelos de color.
        ¿Con qué modelo de color se segmenta mejor?
        Guarda la imagen en/out/practica03/fig_test13_nombrexx.png
        Respuesta:
    """


if __name__ == "__main__":
    do_test09()
    # do_test10()
    # do_test11()
    # do_test12()
