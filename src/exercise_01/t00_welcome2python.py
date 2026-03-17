__author__ = "Artzai"

"""
   TODO: Entiende este código y haz que funcione asegurándote de que tienes bien elegido el intérprete 
   Te enseña ausar lo básico de python y numpy
"""
import matplotlib.pyplot as plt

# cuidado con esto hola
# cambio remoto
# vaya liada
# Para escribir menos puedo poner alias a lo que importo:
import numpy
import numpy as np
import scipy

# Aquí importo las librerías que me resultan útiles
import skimage
import sklearn

# Operaciones basicas de Python

# un comentario se hace con almohadilla

# 1 - Visualizar textos
print("Esto es un texto de python")  # comilla simple
print("Esto tambien es un texto de python")  # doble comilla
num1 = 1
num2 = 0.5
print(
    f"Esto tambien es un texto de python con el número entero {num1} y el número real {num2} metido en la cadena"
)  # doble comilla

# Numpy nos permite trabajar con arrays y matrices, de forma muy parecida a Matlab
matriz1 = np.ones((8, 8))
A = np.uint8(matriz1)
print(A)
B = np.uint8(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
print(B)

# Podemos acceder a todas las filas o columnas de una matriz mediante el operador :
A[0, :] = 0
A[:, 0] = 0
A[:, 7] = 0
A[:, 3] = 2
A[7, :] = 0
print(A)

# Podemos acceder a subregiones de la matriz accediendo mediante el operador corchete
A[2:5, 3:6] = 4
print(A)

# Tambien podemos crear matrices 3D, perfectas para guardar imágenes
ancho_imagen = 100
alto_imagen = 100
imagen = np.zeros((alto_imagen, ancho_imagen, 3))
print(imagen.shape)

# Python es muy limpio, para indicar el alcance de una función no se usan ni begin-end
# ni llaves, todo se indenta de forma ordenada

if imagen.shape[0] > 10:
    print("La imagen es suficientemente grande")
    print("esto está dentro del if")
else:
    print("La imagen es pequeña")
print("esto no está dentro del else porque no está indentado")

# Este for cuenta de cero al nueve
for i in range(10):
    print(i)

milista = ["perro", "gato", "elefante"]
for elemento in milista:
    print(elemento)

for i, elemento in enumerate(milista):
    print(i)
    print(elemento)

for i in range(len(milista)):
    print(i)
    print(milista[i])

# prueba
