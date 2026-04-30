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

from src.exercise_01.t01_load_image import visualizar_imagen
from pathlib import Path

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)

file_histo = "./data/histograma/cerebro_1.jpg"
file_thres = "./data/histograma/bookpage.jpg"
file_7seg = "./data/histograma/Segmentos7.jpg"

file_mariposa = "./data/morfologicos/mariposa.jpg"
file_mariposa_noisy = "./data/morfologicos/mariposa_noisy.jpg"

file_sudoku = "./data/morfologicos/sudoku.jpg"
file_windows = "./data/morfologicos/windows.jpg"

file_bodegon = "./data/color/bodegon.png"


def do_test05():
    """
    NOMBRE:
    TODO:
        Comentar qué hace el código.
        Ver el efecto de variar el valor de threshold en la imagen
        Guardar la mejor imagen resultante en: /out/practica03/fig_test05_nombrexx.png
        Respuesta:
    """

    img_in = skimage.io.imread(file_7seg, as_gray=True)
    # Nos quedamos sólo con canal 0
    img_in = img_in[:, :, 0]
    visualizar_imagen(img_in, "7 segmentos")

    # 1. Manual
    th_value = 10
    binary_image = img_in > th_value
    visualizar_imagen(binary_image, "7 segmentos binarizada con threshold %.2f" % th_value)


def do_test06():
    """
    NOMBRE:
    TODO:
        Comentar qué hace el código.
        Ver el efecto de variar el valor de threshold en la imagen
        Guardar la mejor imagen resultante en: /out/practica03/fig_test06_nombrexx.png
        Respuesta:
    """

    img_in = skimage.io.imread(file_7seg, as_gray=True)
    # Nos quedamos sólo con canal 0
    img_in = img_in[:, :, 0]
    visualizar_imagen(img_in, "7 segmentos")

    # 2. Metodo Otsu:

    th_value_otsu = threshold_otsu(img_in)

    binary_image = img_in > th_value_otsu
    visualizar_imagen(
        binary_image, "7 segmentos binarizada con threshold otsu %.2f" % th_value_otsu
    )


def do_test07():
    """
    NOMBRE:
    TODO:
        Comentar qué hace el código.
        Ver el efecto de variar el valor de threshold en la imagen
        Visualiza y guarda la imagen contenida en local_thres. ¿Qué representa esta imagen?
        Guardar la mejor imagen resultante en: /out/practica03/fig_test07_nombrexx.png
        Respuesta:
    """

    img_in = skimage.io.imread(file_7seg, as_gray=True)
    # Nos quedamos sólo con canal 0
    img_in = img_in[:, :, 0]
    visualizar_imagen(img_in, "7 segmentos")

    # # 3. Metodo Threshold local:
    local_thresh = threshold_local(img_in, block_size=101, offset=10)
    binary_local = img_in > local_thresh
    visualizar_imagen(binary_local, "7 segmentos binarizada con threshold local")


def do_test08():
    """
    NOMBRE:
    TODO:
        Con la imagen sudoku.jpg y los métodos anteriores conseguir segmentar la imagen lo mejor posible para separar dígitos de fondo.
        Explicar las ventajas e inconvenientes de cada método.
        Guardar la mejor imagen resultante en: /out/practica03/fig_test08_nombrexx.png
        Respuesta:
    """


if __name__ == "__main__":
    do_test05()
    # do_test06()
    # do_test07()
    # do_test08()
