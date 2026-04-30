# -*- coding: utf-8 -*-
import scipy.ndimage
import skimage
import skimage.filters

from src.exercise_01.t01_load_image import visualizar_imagen, visualizar_imagenes
from pathlib import Path
__author__ = 106360

import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt

file_histo = "./data/histograma/cerebro_1.jpg"
file_thres = "./data/histograma/bookpage.jpg"
file_7seg = "./data/histograma/Segmentos7.jpg"

file_mariposa = "./data/morfologicos/mariposa.jpg"
file_mariposa_noisy = "./data/morfologicos/mariposa_noisy.jpg"

file_sudoku = "./data/morfologicos/sudoku.jpg"
file_windows = "./data/morfologicos/windows.jpg"

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)


def add_noise_to_image(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.01
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def do_test_efectos_filtros_lineales():
    # Cargamos la imagen
    img_in = skimage.io.imread(file_mariposa_noisy)
    # Pasamos a nivel de gris (se puede aplicar también en color o en el canal L)
    img_gray = skimage.color.rgb2gray(img_in)

    # Filtrado de ruido de media cuadrado
    filtro_A = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    filtro_A = filtro_A / (filtro_A[:]).sum()
    # Filtrado de ruio de media de disco
    filtro_B = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    filtro_B = filtro_B / (filtro_B[:]).sum()

    # Filtro de extracción de borde vertical (Gy)
    filtro_C = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])

    visualizar_imagenes(
        [filtro_A, filtro_B, filtro_C],
        ("filtro A", "Filtro B", "Filtro C"),
        3,
        1,
        rescale_colors=False,
    )

    image_fA = scipy.ndimage.correlate(img_gray, filtro_A)
    image_fB = scipy.ndimage.correlate(img_gray, filtro_B)
    image_fC = scipy.ndimage.correlate(img_gray, filtro_C)

    visualizar_imagenes(
        [img_gray, image_fA, image_fB, image_fC],
        ("Original", "filtro A", "Filtro B", "Filtro C"),
        4,
        1,
        rescale_colors=False,
    )


def do_test_restauracion_imagen():
    """
    Skimage dispone de funciones de filtrado que directamente permiten aplicar lso filtros deseados a las imágenes
    http://scikit-image.org/docs/dev/api/skimage.filters.html
    NOMBRE:
    TODO:
        Aplicar a la imagen de la mariposa con ruido distintos filtros.
        Aplicad a la imagen anterior filtros gausianos y de mediana, visualizad el resultado
        y elegid aquellos filtros que mejoren la imagen.
        Probad diferentes tamaños de filtros y analizad los resultados.
        ¿Qué filtros de imagen funcionan mejor? , ¿Cuál es la causa?
        Visualizar y guardar la imagen en /data/out/practica04/fig_restauracion_imagen_XXX.png
        Respuesta:
    """

    pass


def do_test_afilado_imagen():
    """
    Combinando filtros gausianos a diferentes frecuencias se pueden realizar tareas de enfatizado.
    Este enfatizado se realiza extrayendo las altas frecuencias de la imagen y añadiendo estas altas frecuencias (bordes)
    a la imagen original.
    Revisa el cuaderno de prácticas o las presentaciones de teoría y:

    NOMBRE:
    TODO:
        Selecciona una imagen que se encuentre un poco borrosa.
        Realiza las combinaciones de filtros necesarias para realizar un afilado.
        Explica el proceso.
        Visualizar y guardar la imagen en /data/out/practica04/fig_afilado_imagen_XXX.png
        Respuesta:
    """
    pass


def do_test_gradiente_imagen():
    """
    Otra topología de filtros sirven para la detección de bordes.
    Uno de los filtros más conocidos es el filtro de sobel.
    NOMBRE:
    TODO:
        Para la imagen sudoku
        Calcula y visualiza el gradiente horizontal y vertical de la imagen mediante el filtro de sobel.
        Calcula y visualiza el módulo del gradiente y el argumento del gradiente de la imagen.
        Utilizad la función: scipy.ndimage.convolve()
        Almacena las imágenes generadas en:
        /data/out/practica04/fig_gradiante_imagen_XXX.png
        Respuesta:


    """

    # Cargamos la imagen
    img_in = skimage.io.imread(file_sudoku, as_gray=True)
    visualizar_imagen(img_in, "Imagen Original")

    print(skimage.filters.HSOBEL_WEIGHTS)
    print(skimage.filters.VSOBEL_WEIGHTS)


def do_test_canny():
    """
    Existen métodos más avanzados para la obtención de bordes.
    Por ejemplo el filtro de canny, ampliamente utilizado en
    visión artificial para la localización de bordes.
    NOMBRE:
    TODO:
        Para la imagen sudoku aplicar el filtro de skimage canny
        Analizar el efecto aplicando varios parámetros.
        Obtener la imagen que mejor aísle los bordes de la imagen y almacenarla en:
        /data/out/practica04/fig_canny_XXX.png
        Respuesta:
    """

    # Cargamos la imagen
    img_in = skimage.io.imread(file_sudoku, as_gray=True)
    visualizar_imagen(img_in, "Imagen Original")


def do_test_frangi():
    """
    Existen otros métodos más avanzados para la obtención de bordes.
    Por ejemplo el filtro de frangi, ampliamente utilizado en imagen médica.

    NOMBRE:
    TODO:
        Para la imagen de color deseada aplicar el filtro de skimage frangi
        Analizar el efecto aplicando varios parámetros.
        Obtener la imagen que mejor aísle los bordes de la imagen y almacenarla en:
        /data/out/practica04/fig_frangi_XXX.png
        Respuesta:
    """

    # Cargamos la imagen
    img_in = skimage.io.imread("MIFICHERO", as_gray=True)
    visualizar_imagen(img_in, "Imagen Original")


if __name__ == "__main__":
    # do_test_efectos_filtros_lineales()
    # do_test_restauracion_imagen()
    # do_test_afilado_imagen()

    do_test_gradiente_imagen()
