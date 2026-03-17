__author__ = "mi_nombre_aqui"
# Estas dos líneas inferiores se usan para crear gráficos interactivos
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io

output_folder = (
    Path(__file__).resolve().parent.parent.parent / "outs" / Path(__file__).resolve().parent.name
)
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
    path = Path(nombre_fichero)
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero: {nombre_fichero}")
    imagen_rgb = skimage.io.imread(str(path))
    return imagen_rgb


def guardar_imagen(nombre_fichero, imagen_rgb):
    """
    Guarda una imagen y crea el subdirectorio si no existe.
    :param nombre_fichero: Nombre del fichero
    :param imagen_rgb: imagen numpy array para guardar
    :return:
    """
    # Crea el directorio si no existe
    path = Path(nombre_fichero)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    # Guarda la imagen
    skimage.io.imsave(str(path), imagen_rgb)


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

    fig, ax = plt.subplots(1, 1)
    if imagen.ndim == 2:
        if rescale_colors:
            vmin = 0
            vmax = 1
            if np.ravel(imagen).max() > 1.0:
                vmax = 255
            ax.imshow(imagen, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            ax.imshow(imagen, cmap="gray")
    else:
        ax.imshow(imagen)
    ax.set_title(titulo)
    if save_figure:
        figure_path = Path(figure_save_path)
        if not figure_path.parent.exists():
            figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(figure_path), dpi=600)
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
    if n_row < 1 or n_col < 1:
        raise ValueError("n_row and n_col must be >= 1")
    fig, ax = plt.subplots(n_row, n_col, squeeze=False)

    for idx, (imagen, titulo) in enumerate(zip(lista_imagen, lista_titulos)):
        r = idx // n_col
        c = idx % n_col
        if r >= n_row:
            break
        try:
            if imagen.ndim == 2:
                if rescale_colors:
                    ax[r, c].imshow(imagen, cmap="gray")
                else:
                    ax[r, c].imshow(imagen, cmap="gray")
            else:
                ax[r, c].imshow(imagen)
            ax[r, c].set_title(titulo)
        except Exception:
            pass

    if save_figure:
        figure_path = Path(figure_save_path)
        if not figure_path.parent.exists():
            figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(figure_path), dpi=600)
    plt.show(block=block)


# Es buena costumbre meter este if al final para evitar que se ejecute código al importar este script desde otro.
if __name__ == "__main__":
    output_folder = (
        Path(__file__).resolve().parent.parent.parent
        / "outs"
        / Path(__file__).resolve().parent.name
    )
    output_folder.mkdir(exist_ok=True, parents=True)

    file_path = Path("./data/underwater/Ancuti01.png")
    try:
        imagen_rgb = cargar_imagen(file_path)
        pixel_info = imagen_rgb[20, 30, :]
        image_file_path = output_folder / "image_out.png"
        visualizar_imagen(
            imagen_rgb,
            titulo=f"El valor RGB del pixel (20,30) es {pixel_info[0]},{pixel_info[1]},{pixel_info[2]}",
            save_figure=True,
            figure_save_path=str(image_file_path),
        )
    except FileNotFoundError:
        print(
            f"Fichero de ejemplo no encontrado: {file_path}. Ejecuta desde la raíz del repo o coloca el archivo."
        )
    print("fin")
