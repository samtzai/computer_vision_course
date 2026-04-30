# -*- coding: utf-8 -*-
__author__ = 106360

"""
Leed el documento asociado sobre deep learning (o la charla de Aitor Álvarez) y haced lo siguiente:
- Entrenar un modelo cifar10 con el script que os hemos dado (script T01_train_CNN.py)
- Crear un programita que cargue ese modelo y prediga si en una imagen hay cierta clase:
    Para ello debéis:
        - Cargar el fichero H5 proporcionado por el otro script.
        - Poner la imagen al tamaño (el mismo en el que se entrena la red) y escala (rango de los canales RGB) requerido.
        - Llamar a la función predict del modelo de keras para predecir.

"""
import numpy
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf

from pathlib import Path

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    mi_image_file = "avion.jpg"  # buscad un fichero de imagen de alguna de las clases que existen
    model = tf.keras.models.load_model("mi_modelo.h5")
    # Lo cargamos
    image_rgb = skimage.io.imread(mi_image_file).astype(float) / 255.0
    # Lo recortamos a 32x32
    image_rgb_32x32 = skimage.transform.resize(image_rgb, (32, 32), preserve_range=True)
    imagen_extendida = numpy.expand_dims(image_rgb_32x32, axis=0)
    print(imagen_extendida.shape)
    prediction = model.predict(imagen_extendida)
    print(prediction)
