__author__ = "mi_nombre_aqui"
# Estas dos líneas inferiores se usan para crear gráficos interactivos
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Tkagg")
import matplotlib.pyplot as plt

# Para escribir menos puedo poner alias a lo que importo:
import numpy
import numpy as np
import scipy

# Aquí importo las librerías que me resultan útiles
import skimage
import skimage.io
import sklearn

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)


"""
Date = 17/07/2018
Author = 106376
Project = Práctica 0, principios de python

"""

"""
   En este código proporcionamos herramientas que nos van a permitir cargar, guardar y visualizar 
   imágenes. Lo usaremos durante todas las prácticas.
   
   TODO: Entiende este código y haz que funcione asegurándote de que tienes bien elegido el intérprete    
"""


# Las funciones de python se definen con def. El contenido se escribe indentado:
def cargar_imagen(nombre_fichero):
    """
    Carga una imagen a partir de un fichero
    :param nombre_fichero: Nombre del fichero
    :return:
    """
    imagen_rgb = skimage.io.imread(nombre_fichero)
    return imagen_rgb


def guardar_imagen(nombre_fichero, imagen_rgb):
    """
    Guarda una imagen y crea el subdirectorio si no existe.
    :param nombre_fichero: Nombre del fichero
    :param imagen_rgb: imagen numpy array para guardar
    :return:
    """
    # Crea el directorio si no existe
    folder, filename = os.path.split(nombre_fichero)
    try:
        os.makedirs(folder)
    except Exception:
        pass
    # Guarda la imagen
    skimage.io.imwrite(nombre_fichero, imagen_rgb)


def visualizar_imagen(
    imagen,
    titulo="nombre_del_estudiante",
    block=True,
    save_figure=False,
    figure_save_path="out/fig_sample.png",
    rescale_colors=True,
):
    """
    Esta función visualiza y almacena una imagen y salva la figura si se indica
    :param imagen_rgb: imagen RGB de tipo numpy array
    :param titulo: Indica el nombre del estudiante más la información adicional requerida
    :param block: Permite que al visualizar la imagen en programa pare hasta cerrar la ventana
    :return:
    """

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    if len(imagen.shape) == 2:
        if rescale_colors:
            vmin = 0
            vmax = 1
            if np.ravel(imagen).max() > 1.0:
                vmax = 255
            ax.imshow(imagen, cmap=plt.get_cmap("gray"), vmin=vmin, vmax=vmax)
        else:
            ax.imshow(imagen)

    else:
        ax.imshow(imagen)
    ax.set_title(titulo)
    if save_figure:
        folder, filename = os.path.split(figure_save_path)
        try:
            os.makedirs(folder)
        except Exception:
            pass
        plt.savefig(figure_save_path, dpi=600)
    plt.show(block=block)


def visualizar_imagenes(
    lista_imagen,
    lista_titulos,
    n_row,
    n_col,
    block=True,
    save_figure=False,
    figure_save_path="out/fig_sample.png",
    rescale_colors=True,
):
    """
    Esta función visualiza y almacena una imagen y salva la figura si se indica
    :param imagen_rgb: imagen RGB de tipo numpy array
    :param titulo: Indica el nombre del estudiante más la información adicional requerida
    :param block: Permite que al visualizar la imagen en programa pare hasta cerrar la ventana
    :return:
    """
    if n_row < 2:
        raise Exception("n_row tiene que ser mayor que 2")
    fig, ax = plt.subplots(n_row, n_col, sharex=True, sharey=True)

    nc = 0
    nr = 0
    if n_col > 1:
        for i, (imagen, titulo) in enumerate(zip(lista_imagen, lista_titulos, strict=True)):
            try:
                if len(imagen.shape) == 2:
                    if rescale_colors:
                        ax[nr, nc].imshow(imagen, cmap=plt.get_cmap("gray"))
                    else:
                        ax[nr, nc].imshow(imagen, cmap=plt.get_cmap("gray"))
                else:
                    ax[nr, nc].imshow(imagen)
                ax[nr, nc].set_title(titulo)
            except Exception:
                pass
            nr += 1
            if nr == n_row:
                nr = 0
                nc += 1
    else:
        for i, (imagen, titulo) in enumerate(zip(lista_imagen, lista_titulos, strict=True)):
            try:
                if len(imagen.shape) == 2:
                    ax[nr].imshow(imagen, cmap=plt.get_cmap("gray"))
                else:
                    ax[nr].imshow(imagen)
                ax[nr].set_title(titulo)
            except Exception:
                pass
            nr += 1
            if nr == n_row:
                nr = 0
                nc += 1

    if save_figure:
        folder, filename = os.path.split(figure_save_path)
        try:
            os.makedirs(folder)
        except Exception:
            pass
        plt.savefig(figure_save_path, dpi=600)
    plt.show(block=block)


# Es buena costumbre meter este if al final para evitar que se ejecute código al importar este script desde otro.
if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    file_path = "./data/underwater/Ancuti01.png"
    imagen_rgb = cargar_imagen(file_path)
    pixel_info = imagen_rgb[20, 30, :]
    image_file_path = output_folder / "image_out.png"
    visualizar_imagen(
        imagen_rgb,
        titulo=f"Artzai, El valor RGB del pixel (20,30) es {pixel_info[0]},{pixel_info[1]},{pixel_info[2]}",
        save_figure=True,
        figure_save_path=image_file_path,
    )
    print("fin")
