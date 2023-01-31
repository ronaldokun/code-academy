# Aritmética

# Para executar este script:
# python cap02-23-opencv-arithmetic.py --image images/ferrari.jpg
 
# Imports
import numpy as np
import argparse
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho para a imagem")
args = vars(ap.parse_args())

# Carregando a imagem
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# As imagens são matrizes NumPy, armazenadas como números inteiros de 8 bits (uint8).
# Isso significa que os valores de nossos pixels estarão na faixa [0, 255]. 
# Ao usar funções como cv2.add e cv2.subtract, podemos adicionar ou remover intensidade aos pixels.
# Mas atenção: essas funções tem comportamento diferente do que operações simples entre os elementos da matriz

# Vamos aumentar a intensidade de todos os pixels em nossa imagem por 100. 
# Realizamos isso construindo uma matriz NumPy que tem o mesmo tamanho da nossa matriz
# e multiplicando por 100 para criar uma matriz preenchida
# com 100's. Então, simplesmente adicionamos as imagens juntas. 
# Esta imagem será mais brilhante
M = np.ones(image.shape, dtype = "uint8") * 100
added = cv2.add(image, M)
cv2.imshow("Added", added)

# Da mesma forma, podemos subtrair 50 de todos os pixels em nossa imagem e torná-la mais escura:
M = np.ones(image.shape, dtype = "uint8") * 50
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)