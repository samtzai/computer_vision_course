__author__ = "Artzai"

"""Exercise 01: Welcome to Python.

Small demo showing basic Python and NumPy usage for the course.
"""

import numpy as np


def main():
    # 1 - Visualizar textos
    print("Esto es un texto de python")
    print("Esto tambien es un texto de python")
    num1 = 1
    num2 = 0.5
    print(f"Ejemplo: entero={num1}, real={num2}")

    # Numpy: arrays y matrices
    matriz1 = np.ones((8, 8), dtype=np.uint8)
    A = matriz1.copy()
    print("Matriz A (8x8):")
    print(A)

    B = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    print("Matriz B (3x3):")
    print(B)

    # Acceder y modificar filas/columnas/subregiones
    A[0, :] = 0
    A[:, 0] = 0
    A[:, 7] = 0
    A[:, 3] = 2
    A[7, :] = 0
    A[2:5, 3:6] = 4
    print("Matriz A tras modificaciones:")
    print(A)

    # Matrices 3D (imágenes)
    ancho_imagen = 100
    alto_imagen = 100
    imagen = np.zeros((alto_imagen, ancho_imagen, 3), dtype=np.uint8)
    print("Imagen shape:", imagen.shape)

    # Control de flujo
    if imagen.shape[0] > 10:
        print("La imagen es suficientemente grande")
    else:
        print("La imagen es pequeña")

    # Bucles y listas
    print("Contando de 0 a 9:")
    for i in range(10):
        print(i)

    milista = ["perro", "gato", "elefante"]
    for elemento in milista:
        print(elemento)

    for i, elemento in enumerate(milista):
        print(i, elemento)


if __name__ == "__main__":
    main()

