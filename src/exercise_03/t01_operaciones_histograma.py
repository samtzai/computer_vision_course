# -----------------------
#    Date = 17/07/2018
#  Author = 106376
# Project = code
# -----------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.exposure
import skimage.io

from src.exercise_01.t01_load_image import visualizar_imagen

file_histo = "./data/histograma/cerebro_1.jpg"
file_thres = "./data/histograma/bookpage.jpg"
file_7seg = "./data/histograma/Segmentos7.jpg"

file_mariposa = "./data/morfologicos/mariposa.jpg"
file_mariposa_noisy = "./data/morfologicos/mariposa_noisy.jpg"

file_sudoku = "./data/morfologicos/sudoku.jpg"
file_windows = "./data/morfologicos/windows.jpg"

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)


def do_test01():
    # Calcular un histograma
    """
    NOMBRE:
    TODO:
        Comentar qué se observa en la distribución de histograma.


    """
    img_in = skimage.io.imread(file_histo, as_gray=True)
    visualizar_imagen(img_in, "Cerebro")

    # Con numpy
    hist, bins = np.histogram(img_in.ravel(), 256, [0, 1.0])
    plt.plot(hist)
    plt.savefig(output_folder / "fig_test01_histograma.png")
    plt.show()

    # Con matplotlib
    plt.hist(img_in.ravel() * 256, 256, [0, 256], color="r")
    plt.xlim([0, 256])
    plt.show()


def do_test02():
    """
    NOMBRE:
    TODO:
        Comentar qué se observa en la distribución de histograma en cada uno de los pasos.
    """

    # Estirar / ecualizar un histograma
    # Tiempo estimado -> 10 minutos (total = 15 min)
    img_in = skimage.io.imread(file_histo)
    if len(img_in.shape) == 3:
        # Convertir a gris
        img_in = np.average(img_in, axis=-1)

    # Stretching
    # Io = (Ii-Mini)*(((Maxo-Mino)/(Maxi-Mini))+Mino)
    min_in = img_in.min()
    max_in = img_in.max()
    min_out = 0
    max_out = 255
    img_strecthed = (img_in - min_in) * (((max_out - min_out) / (max_in - min_in)) + min_out)

    # Stretching using skimage

    img_strecthed2 = skimage.exposure.rescale_intensity(img_in, out_range=(0, 255))

    # Ecualizacion de histograma

    img_equ = (skimage.exposure.equalize_hist(img_in) * 255).astype("uint8")

    # Ecualizacion de histograma adaptativa
    img_adapt_equ = skimage.exposure.equalize_adapthist(
        img_in.astype(float) / 255, kernel_size=None, clip_limit=0.01, nbins=256
    )

    plt.subplot(131)
    plt.hist(img_in.ravel(), 256, [0, 256], color="b")
    plt.subplot(132)
    plt.hist(img_strecthed2.ravel(), 256, [0, 256], color="r")
    plt.subplot(133)
    plt.hist(img_equ.ravel(), 256, [0, 256], color="g")
    plt.show(block=True)

    plt.figure()
    plt.subplot(141)
    plt.imshow(img_in, cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
    plt.subplot(142)
    plt.imshow(img_strecthed2, cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
    plt.subplot(143)
    plt.imshow(img_equ, cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
    plt.subplot(144)
    plt.imshow(img_adapt_equ, cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
    plt.show(block=True)


def do_test03():
    """
    NOMBRE:
    TODO:
        Repetir el ejercicio do_test02 con la imagen sudoku.jpg :
        Extraer y visualizar los histogramas de la imagen original
        de la imagen con el histograma estirado y de la imagen ecualizada.
        Analizar el uso de CLAHE (imadapthist) en estas imágenes.
        Comentar resultados
        Almacenar las figuras resultantes en el directorio /out/practica03/fig_test03_nombrexx.png
        Respuesta:
    """


def do_test04():
    """
    NOMBRE:
    TODO:
        Completar el ejercicio do_test02 con la imagen médica de vuestra elección :
        Extraer y visualizar los histogramas de la imagen original
        de la imagen con el histograma estirado y de la imagen ecualizada.
        Analizar el uso de CLAHE (imadapthist) en estas imágenes.
        Comentar resultados
        Almacenar las figuras resultantes en el directorio /out/practica03/fig_test04_nombrexx.png
        Respuesta:
    """


if __name__ == "__main__":
    do_test01()
    do_test02()
    do_test03()
    do_test04()

    input()
