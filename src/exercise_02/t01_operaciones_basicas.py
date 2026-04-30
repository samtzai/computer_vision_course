# -*- coding: utf-8 -*-
__author__ = "adios"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io

from src.exercise_01.t01_load_image import visualizar_imagen

"""
TODO: Elige una imagen para usar
"""
file_img_elegida_estudiante = ""
file_img_01 = "./data/underwater/Ancuti01.png"
file_img_02 = "./data/underwater/Ancuti03.png"
file_logo = "./data/mascaras/logo_ehu.png"
file_base = "./data/mascaras/orto.jpg"

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)


def do_test01():
    """
    NOMBRE: Artzai
    TODO:
        COMENTA LO QUE HACE EL CÓDIGO
        Respuesta:
        XXXXX
    """

    img_in = skimage.io.imread(file_img_01)
    print("Dimensiones: " + str(img_in.shape))
    print("Tamanio (numero pixeles x canales): " + str(img_in.size))
    print("Tipo de dato: " + str(img_in.dtype))

    # copiar imagen
    img_out = img_in.copy()
    print("Dimensiones: " + str(img_out.shape))
    print("Tamanio (numero pixeles x canales): " + str(img_out.size))
    print("Tipo de dato: " + str(img_out.dtype))


def do_test02():
    """
    NOMBRE:
    TODO:
        COMENTA LO QUE HACE EL CÓDIGO
        Respuesta:
        XXXXX
    """
    for i in range(1, 11):
        file_img = f"./data/underwater/Ancuti{i:02d}.png"
        print("Image: " + file_img)
        img_in = skimage.io.imread(file_img)
        visualizar_imagen(img_in, f"imagen número {i}", block=True)


def do_test03():
    """
    NOMBRE:
    TODO:
        Sobre la imagen a color elegida por tí: file_img_elegida_estudiante
        Cual es el tamaño de img_in, y de img_out?
        Modifica esta función para que en el título de las figuras aparezca también el tamaño de la imagen (ancho, alto, número de canales) y fuardalo en el directorio out de la práctica.
        Respuesta:
        XXXXX
    """
    # Cargar una imagen a color, convertir a gris y guardar

    img_in = skimage.io.imread(file_img_01)
    img_out = skimage.color.rgb2gray(img_in)

    visualizar_imagen(
        img_in,
        titulo="imagen original",
        save_figure=True,
        figure_save_path=output_folder / "fig_test03_in.png",
    )
    visualizar_imagen(
        img_out,
        titulo="imagen gris",
        save_figure=True,
        figure_save_path=output_folder / "fig_test03_out.png",
    )


def do_test04():
    """
    NOMBRE:
    TODO:
        Sobre la imagen a color elegida por tí: file_img_elegida_estudiante
        Elige uno de los tres canales de img_in y ponlo a cero.
        Luego introduce ese canal en img_in.
        Para ello usa slicing, que también se puede utilizar para asignar.
        visualiza y comenta el resultado.
        Almacena la figura resultante
        Respuesta:
        XXXXX
    """

    img_in = skimage.io.imread(file_img_01)

    visualizar_imagen(
        img_in,
        titulo="imagen original",
        save_figure=True,
        figure_save_path=output_folder / "fig_test04_in.png",
    )

    # Con funciones de numpy
    np_b = img_in[:, :, 0]
    np_g = img_in[:, :, 1]
    np_r = img_in[:, :, 2]

    visualizar_imagen(
        np_b,
        titulo="canal r",
        save_figure=True,
        figure_save_path=output_folder / "fig_test04_r.png",
    )
    visualizar_imagen(
        np_g,
        titulo="canal g",
        save_figure=True,
        figure_save_path=output_folder / "fig_test04_g.png",
    )
    visualizar_imagen(
        np_r,
        titulo="canal b",
        save_figure=True,
        figure_save_path=output_folder / "fig_test04_b.png",
    )

    # img_in_un_canal_a_cero = TODO:
    # visualizar_imagen(img_in_un_canal_a_cero, titulo='imagen con un canal eliminado',save_figure=True,figure_save_path=output_folder / 'fig_test04_canal_eliminado.png')


def do_test05():
    """
    NOMBRE:
    TODO:
        Repite el código sobre la imagen a color elegida por tí: file_img_elegida_estudiante
        Almacena la figura resultante
        ¿Qué hace el código?
        Respuesta:
        XXXXX
    """
    # Asignacion de ROIs
    # Tiempo estimado -> 5 minutos (total = 40 min)

    img_in = skimage.io.imread(file_img_01)
    img_roi = img_in[15:350, 180:480, :]
    visualizar_imagen(
        img_roi,
        titulo="imagen ROI",
        save_figure=True,
        figure_save_path=output_folder / "fig_test05_roi.png",
    )


def do_test06():
    """

    NOMBRE:
    TODO:
        ¿Qué hace el código?
        Respuesta:
        XXXXX
    """

    img_in = skimage.io.imread(file_img_01)
    visualizar_imagen(img_in, titulo="imagen original")
    img_roi = img_in[15:350, 180:480, :]
    img_roi_gris = np.average(img_roi, axis=2)  # Calculamos el gris como la media de R, G y B

    img_out = img_in.copy()
    img_out[15:350, 180:480, 0] = img_roi_gris
    img_out[15:350, 180:480, 1] = img_roi_gris
    img_out[15:350, 180:480, 2] = img_roi_gris

    visualizar_imagen(
        img_out,
        titulo="imagen reconstruida",
        save_figure=True,
        figure_save_path=output_folder / "fig_test06_recon.png",
    )


def do_test07():
    """
    NOMBRE:
    TODO:
        Basándote en el ejemplo anterior, cambia la estatua por la del buzo

    """

    img_in = skimage.io.imread(file_img_01)
    visualizar_imagen(img_in, titulo="imagen original")

    # TODO: tu código aquí

    # visualizar_imagen(img_out, titulo='imagen reconstruida', save_figure=True,
    #                   figure_save_path=output_folder / 'fig_test06_recon.png')


def do_test08():
    """
    NOMBRE:
    TODO: Revisa el código y analiza los resultados
        ¿Cuál es el tipo de dato de la imagen?
        Cuál es el valor máximo que puede alcanzar img_suma?
        Qué pasa si convierto las imágenes a números reales y repito el proceso?
        Respuesta:
    """

    img_in1 = skimage.io.imread(file_img_01)
    img_in2 = skimage.io.imread(file_img_02)

    # img_in1 = skimage.io.imread(file_img_01).astype(float)/255.0
    # img_in2 = skimage.io.imread(file_img_02).astype(float)/255.0

    visualizar_imagen(img_in1, titulo="imagen 1")
    visualizar_imagen(img_in2, titulo="imagen 2")

    # suma imagenes
    img_suma = img_in1 + img_in2
    visualizar_imagen(
        img_suma,
        titulo="imagen suma",
        save_figure=True,
        figure_save_path=output_folder / "fig_test08_suma.png",
    )

    # resta imagenes
    img_resta = img_in1 - img_in2
    visualizar_imagen(
        img_resta,
        titulo="imagen resta",
        save_figure=True,
        figure_save_path=output_folder / "fig_test08_resta.png",
    )

    # resta imagenes absoluta
    img_resta_abs = np.abs(img_in1 - img_in2)
    visualizar_imagen(
        img_resta_abs,
        titulo="imagen resta abs",
        save_figure=True,
        figure_save_path=output_folder / "fig_test08_resta_abs.png",
    )

    # Blending
    img_blending = 0.30 * img_in1 + 0.70 * img_in2
    visualizar_imagen(
        img_blending,
        titulo="imagen blending",
        save_figure=True,
        figure_save_path=output_folder / "fig_test08_blending.png",
    )


def do_test09():
    """
    NOMBRE:
    TODO:
        Revisa el código y analiza los resultados presta atención a la máscara,
        la necesitarás para el trabajo con barrio sésamo.
        ¿Cómo afecta el valor threshold?
        Respuesta:

    """
    # Operaciones logicas -> And / Not y mascaras
    # Tiempo estimado -> 20 minutos (total = 110 min)
    # cargar las imagenes del ejemplo

    img_logo = skimage.io.imread(file_logo)
    img_base = skimage.io.imread(file_base)
    visualizar_imagen(img_base, titulo="imagen base")
    visualizar_imagen(img_logo, titulo="imagen logo")

    # Seleccionar la zona de la imagen donde irá el logo
    rows, cols, channels = img_logo.shape
    roi = img_base[0:rows, 0:cols, :]

    # Ojo, rgb2gray convierte la imagen a float y la rerangea de cero a uno!
    img_logo_gris = skimage.color.rgb2gray(img_logo)

    # Mascara de la info del logo
    threshold = 0.2
    mask = img_logo_gris > threshold
    visualizar_imagen(mask, titulo="máscara logo")

    # mascara invertida
    mask_inv = np.logical_not(mask)
    visualizar_imagen(mask_inv, titulo="máscara invertida")

    # Crear las máscaras con tres canales para que soporten RGB y poder multiplicar
    mask_inv = np.stack((mask_inv, mask_inv, mask_inv), axis=-1)
    mask = np.stack((mask, mask, mask), axis=-1)

    # Poner a 0 la zona de la imagen del logo que no nos interesa
    img_logo_limpia = np.logical_and(mask, img_logo) * img_logo
    visualizar_imagen(img_logo_limpia, titulo="logo limpio")

    # Poner a 0 la zona de la imagen base que va a ir a cero
    img_roi_limpia = np.logical_and(mask_inv, roi) * roi
    visualizar_imagen(img_roi_limpia, titulo="logo limpio")

    # Sumar las 2 imagenes
    dst = img_roi_limpia + img_logo_limpia
    visualizar_imagen(dst, titulo="suma")

    img_base[0:rows, 0:cols] = dst
    # visualizar_imagen(img_base, titulo='logo_output',save_figure=True,figure_save_path=output_folder / "fig_test09_mascara.png")


if __name__ == "__main__":
    # do_test01()
    do_test02()
    # do_test03()
    # do_test04()
    # do_test05()
    # do_test06()
    # do_test07()
    # do_test08()
    # do_test09()

    input()
