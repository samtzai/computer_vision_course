# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use("TkAgg")
from skimage.color import rgb2lab

__author__ = "106360"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import skimage
import skimage.feature as ft
import skimage.io
import sklearn

output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
output_folder.mkdir(exist_ok=True, parents=True)


def GetDescriptorLBPUniform(gray_image, radius, n_points, n_bins):
    gray_image = skimage.img_as_float(gray_image)
    radius = 4  # Selected as the best for multiple scales. We need a fixed number here, to have a fixed number of points.
    n_points = 8 * radius
    METHOD = "uniform"  # 'uniform'
    n_bins = 27  # 256#10
    lbp_1 = ft.local_binary_pattern(gray_image[:, :, 0], n_points, radius, METHOD)
    lbp_2 = ft.local_binary_pattern(gray_image[:, :, 1], n_points, radius, METHOD)
    lbp_3 = ft.local_binary_pattern(gray_image[:, :, 2], n_points, radius, METHOD)

    descriptors = numpy.zeros((lbp_1.shape[0], lbp_1.shape[1], 3), dtype="uint8")
    descriptors[:, :, 0] = lbp_1
    descriptors[:, :, 1] = lbp_2
    descriptors[:, :, 2] = lbp_3

    # lbp = ma.masked_array(lbp,mask=numpy.logical_not(mask_array))
    return descriptors


def extract_image_descriptor_lbp(descriptor_lbp, window_size, step):
    M = descriptor_lbp.shape[0]
    N = descriptor_lbp.shape[1]
    M2 = int(numpy.floor(M / step))
    N2 = int(numpy.floor(N / step))
    n_bins = 27
    descriptor = numpy.zeros((M2, N2, n_bins * 3))
    for i in range(M2):
        for j in range(N2):
            ii = i * step
            jj = j * step

            lbp_1_win = descriptor_lbp[ii : ii + window_size, jj : jj + window_size, 0]
            d1 = numpy.histogram(lbp_1_win.squeeze(), n_bins, density=True)
            lbp_2_win = descriptor_lbp[ii : ii + window_size, jj : jj + window_size, 1]
            d2 = numpy.histogram(lbp_2_win.squeeze(), n_bins, density=True)
            lbp_3_win = descriptor_lbp[ii : ii + window_size, jj : jj + window_size, 2]
            d3 = numpy.histogram(lbp_3_win.squeeze(), n_bins, density=True)

            descriptor[i, j, 0:27] = d1[0]
            descriptor[i, j, 27 : 27 + 27] = d2[0]
            descriptor[i, j, 27 + 27 : 27 + 27 + 27] = d3[0]
    return descriptor

    # compressed_lbp = lbp.compressed()
    #
    #
    # fd = numpy.histogram(lbp.compressed(),n_bins,density=True)
    # return fd[0].tolist()


def extract_image_descriptor_mean_var(image_RGB, window_size, step):
    M = image_RGB.shape[0]
    N = image_RGB.shape[1]
    M2 = int(numpy.floor(M / step))
    N2 = int(numpy.floor(N / step))
    n_bins = 3
    descriptor = numpy.zeros((M2, N2, n_bins * 2))
    for i in range(M2):
        for j in range(N2):
            ii = i * step
            jj = j * step

            lbp_1_win = image_RGB[ii : ii + window_size, jj : jj + window_size, 0].squeeze()
            d1 = [numpy.mean(lbp_1_win), numpy.std(lbp_1_win)]
            lbp_2_win = image_RGB[ii : ii + window_size, jj : jj + window_size, 1].squeeze()
            d2 = [numpy.mean(lbp_2_win), numpy.std(lbp_2_win)]
            lbp_3_win = image_RGB[ii : ii + window_size, jj : jj + window_size, 2].squeeze()
            d3 = [numpy.mean(lbp_3_win), numpy.std(lbp_3_win)]

            descriptor[i, j, 0:2] = d1
            descriptor[i, j, 2:4] = d2
            descriptor[i, j, 4:6] = d3
    return descriptor


def segment_unsupervised(image, numclasses, normalize_data):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = numpy.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
    if normalize_data:
        sc = StandardScaler()
        X = sc.fit_transform(X)
    kmeans = KMeans(n_clusters=numclasses, random_state=0).fit(X)

    segmented = kmeans.predict(X)

    return numpy.reshape(segmented, (image.shape[0], image.shape[1]))


def generate_supervised_model(image_descriptor_real_size, selected_points, y_class, normalize_data):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler

    X = numpy.zeros((len(selected_points), image_descriptor_real_size.shape[2]))
    for i, point in enumerate(selected_points):
        X[i, :] = image_descriptor_real_size[int(point[1]), int(point[0]), :]

    sc = None
    if normalize_data:
        sc = StandardScaler()
        X = sc.fit_transform(X)
    # clf = GaussianNB()
    import sklearn.ensemble

    clf = sklearn.ensemble.RandomForestClassifier()
    clf.fit(X, y_class)

    return clf, sc


def segment_supervised(image_descriptor, model, normalize_data, scaler):
    from sklearn.preprocessing import StandardScaler

    X = numpy.reshape(
        image_descriptor,
        (image_descriptor.shape[0] * image_descriptor.shape[1], image_descriptor.shape[2]),
    )
    if normalize_data:
        X = scaler.transform(X)
    probability_image = model.predict_proba(X)
    segmentation = model.predict(X)

    return numpy.reshape(
        probability_image,
        (image_descriptor.shape[0], image_descriptor.shape[1], probability_image.shape[1]),
    ), numpy.reshape(segmentation, (image_descriptor.shape[0], image_descriptor.shape[1]))


if __name__ == "__main__":
    """
    En esta práctica vamos a estudiar cómo es posible extraer descriptores no sólo de forma como en la práctica anterior,
    sino que reflejen el color y la textura de cada zona de la imagen.
    
    Así, a nuestra imagen de tamaño MxNx3 la convertimos en una imagen MxNxK donde K es el número de descriptores que usamos.
    Estos descriptores se diseñan según el problema a resolver. En nuestro caso usamos dos grupos:
    1) Descriptor de color: media y varianza de los canales Lab. Para cada píxel se coge una vecindad y se saca como descriptor su 
    media de Lab y su varianza, lo que permite discriminar el color y cómo varía.
    2) Descriptor LBP un descriptor que permite detectar texturas en imagen
    
    Vamos a analizar el código:
    
    """
    show_loading = True
    show_descriptor = True
    show_lbp_channels = True
    do_unsupervised_clustering = True
    do_supervised_clustering = True

    use_texture_descriptors = True
    use_color_descriptors = True

    # Load image
    imageRGB = skimage.io.imread("./data/color/donosti.jpg")

    window_size = 48
    step = 8

    radius_LBP = 4
    num_LBP = radius_LBP * 8
    num_bins = 33

    M, N, K = imageRGB.shape

    descriptor_lbp = GetDescriptorLBPUniform(imageRGB, radius_LBP, num_LBP, num_bins)

    if show_loading:
        print(
            "Esta imagen muestra los descriptores LBP (Local binary patterns) que indican textura"
        )
        # show image
        fig, ax = plt.subplots(4, 1, sharey=True, sharex=True)
        ax[0].set_xlabel("Original")
        ax[0].imshow(imageRGB)
        ax[1].imshow(descriptor_lbp[:, :, 0], cmap="jet")
        ax[1].set_xlabel("Descriptor LBP 1")
        ax[2].imshow(descriptor_lbp[:, :, 1], cmap="jet")
        ax[2].set_xlabel("Descriptor LBP 2")
        ax[3].imshow(descriptor_lbp[:, :, 2], cmap="jet")
        ax[3].set_xlabel("Descriptor LBP 3")
        print("Cierra la figura para continuar")
        plt.show()

    # Mediante esta función creamos una imagen que contiene los descriptores LBP
    image_texture_descriptor = extract_image_descriptor_lbp(descriptor_lbp, window_size, step)

    # Mediante esta función calculamos los descriptores de color
    image_color_descriptor = extract_image_descriptor_mean_var(
        rgb2lab(imageRGB.copy()), window_size, step
    )

    if show_loading:
        # show image
        print("Esta imagen muestra los descriptores Lab que indican color")
        fig, ax = plt.subplots(3, 2, sharey=True, sharex=True)
        ax[0][0].imshow(image_color_descriptor[:, :, 0] / 255.0)
        ax[0][0].set_xlabel("Media Descriptor L")
        ax[1][0].imshow(image_color_descriptor[:, :, 2] / 255.0)
        ax[1][0].set_xlabel("Media Descriptor a")
        ax[2][0].imshow(image_color_descriptor[:, :, 4] / 255.0)
        ax[2][0].set_xlabel("Media Descriptor b")
        ax[0][1].imshow(image_color_descriptor[:, :, 1] / 255.0)
        ax[0][1].set_xlabel("Desv std Descriptor L")
        ax[1][1].imshow(image_color_descriptor[:, :, 3] / 255.0)
        ax[1][1].set_xlabel("Desv std Descriptor a")
        ax[2][1].imshow(image_color_descriptor[:, :, 5] / 255.0)
        ax[2][1].set_xlabel("Desv std Descriptor b")
        print("Cierra la figura para continuar")
        plt.show()

    # color
    if use_texture_descriptors == False:
        image_descriptor = image_color_descriptor
    if use_color_descriptors == False:
        image_descriptor = image_texture_descriptor
    if (use_texture_descriptors == False) and (use_color_descriptors == False):
        raise Exception("Algo hay que usar no?")
    if (use_texture_descriptors == True) and (use_color_descriptors == True):
        image_descriptor = numpy.dstack([image_color_descriptor, image_texture_descriptor])

    image_descriptor_real_size = skimage.transform.resize(
        image_descriptor, (M, N), order=0, preserve_range=True
    )

    print("Vamos a observar la apariencia de los descriptores en varios puntos de la imagen.")
    plt.close("all")
    plt.imshow(imageRGB)
    plt.show(block=False)

    # Plot descriptors of different places
    if show_descriptor:
        print(
            "Ahora seleciona 6 puntos de los que quieras ver los descriptores de la imagen original"
        )
        points = plt.ginput(6, timeout=-1)
        fig2, ax2 = plt.subplots(6, 1, sharex=True, sharey=True)
        for i, point in enumerate(points):
            x = int(point[0])
            y = int(point[1])
            ax2[i].plot(image_descriptor_real_size[y, x, :])
            ax2[i].set_xlabel("Point %d" % i)
        print("Cierra la figura para continuar")
        plt.show(block=True)

    if do_unsupervised_clustering:
        print("Ahora vamos a generar varios clusteres no supervisados")
        segmentation = segment_unsupervised(image_descriptor, numclasses=8, normalize_data=True)
        segmentation_real_size = skimage.transform.resize(segmentation, (M, N), order=0)
        fig3, ax3 = plt.subplots(2, 1, sharex=True, sharey=True)
        ax3[0].imshow(imageRGB)
        ax3[0].set_xlabel("original")
        ax3[1].imshow(segmentation_real_size)
        ax3[1].set_xlabel("segmented (unsupervised)")
        print("Cierra la figura para continuar")
        plt.show(block=True)

    if do_supervised_clustering:
        print("Ahora vamos a aprender un modelo supervisado")
        n_element_per_class = 10
        n_classes = 4

        fig4, ax4 = plt.subplots(1, 1, sharex=True, sharey=True)
        ax4.imshow(imageRGB)
        ax4.set_xlabel("original")

        selected_points = []
        y_class = []
        for i in range(n_classes):
            print("Selecciona %d puntos para definir la clase %d " % (n_element_per_class, i))
            points = plt.ginput(n_element_per_class, timeout=-1)
            selected_points = selected_points + points
            y_class = y_class + [i] * n_element_per_class

        fig3, ax3 = plt.subplots(3, 1, sharex=True, sharey=True)
        ax3[0].imshow(imageRGB)
        ax3[0].set_xlabel("original")
        normalize_data = True

        # Generamos un modelo con scikit-learn
        model, sc = generate_supervised_model(
            image_descriptor_real_size, selected_points, y_class, normalize_data
        )

        probability_image, segmentation = segment_supervised(
            image_descriptor, model, normalize_data, sc
        )
        segmentation_real_size = skimage.transform.resize(segmentation, (M, N), order=0)
        probability_image_real_size = skimage.transform.resize(probability_image, (M, N), order=0)
        ax3[1].imshow(segmentation_real_size)
        ax3[1].set_xlabel("segmented (supervised)")
        ax3[2].imshow(probability_image_real_size[:, :, 0])
        ax3[2].set_xlabel("Probability map")
        print("Cierra la figura para continuar")
        plt.show(block=True)

    print("Cierra la figura para finalizar")
    plt.show()
