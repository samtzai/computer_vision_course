# -*- coding: utf-8 -*-
__author__ = 106360

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import skimage.io
from skimage import data
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

from src.exercise_01.t01_load_image import visualizar_imagen, visualizar_imagenes

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)
file_geometrias = "./data/color/geometrias.png"


def do_test_image_figuras():
    """
    NOMBRE:
    TODO:
        Ejecuta el script y analiza el listado de descriptores que se obtienen.

    """

    image_rgb = skimage.io.imread(file_geometrias)

    visualizar_imagen(image_rgb, "Imagen_Original")

    image_gray = skimage.color.rgb2gray(image_rgb)
    image_binary = image_gray > 0.1
    image_binary_filtered = skimage.morphology.binary_closing(image_binary)
    visualizar_imagen(image_binary_filtered, "Imagen binaria")

    # label image regions
    label_image = label(image_binary_filtered)
    image_label_overlay = label2rgb(label_image, image=image_rgb)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    lista_descriptores = []
    for region in regionprops(label_image):
        if region.area >= 10:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle(
                (minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor="red", linewidth=2
            )
            ax.add_patch(rect)

            lista_descriptores.append(
                [region.label, region.area, region.extent, region.solidity, region.eccentricity]
            )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    # Pandas es una librería para manejo de datos.
    pandas_dataframe_descriptores = pd.DataFrame.from_records(
        lista_descriptores,
        columns=["Label", "Area", "Extent", "Solidity", "Eccentricity"],
        index="Label",
    )

    print(pandas_dataframe_descriptores)


def do_test_image_figuras_2():
    """
    NOMBRE:
    TODO:
        Modifica el script anterior de tal forma que:
        - Las estrellas les pinte un recuadro rojo
        - Los círculos y cuadrados les pinte un recuadro verde.
        Almacena las imágenes en /data/out/practica05/fig_feature_extraction_xxx.png
        Comenta el método empleado:
        Respuesta:
    """

    pass


if __name__ == "__main__":
    do_test_image_figuras()
