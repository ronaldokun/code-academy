# Shift

# Para executar este script:
# python cap02-19-opencv-shift.py --image images/ferrari.jpg

# Imports
import numpy as np
import argparse
import utils
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho para a imagem")
args = vars(ap.parse_args())

# Carregando a imagem
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Shifting de uma imagem é dada por uma matriz NumPy na forma:
#	[[1, 0, shiftX], [0, 1, shiftY]]

# Você simplesmente precisa especificar quantos pixels você deseja para mudar a imagem nas direções X e Y.
# Vamos fazer o shift da imagem 25 pixels para a direita e 50 pixels para baixo.
M = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Shift para baixo e para direita", shifted)

# Agora, vamos mudar a imagem 50 pixels para a esquerda e 90 pixels para cima. 
# Realizamos isso usando valores negativos.
M = np.float32([[1, 0, -50], [0, 1, -90]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Shift para cima e para esquerda", shifted)

# Finalmente, vamos usar nossa função auxiliar em utils.py para deslocar a imagem para baixo 100 pixels.
shifted = utils.translate(image, 0, 100)
cv2.imshow("Shift para baixo", shifted)
cv2.waitKey(0)