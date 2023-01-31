# Rotação

# Para executar este script:
# python cap02-19.1-opencv-rotation.py --image images/ferrari.jpg
 
# Imports
import numpy as np
import argparse
import utils
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho para a imagem")
args = vars(ap.parse_args())

# Carrega a imagem
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Obtém as dimensões da imagem e calcule o centro da imagem
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# Rotaciona a imagem em 45 graus
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotacionado em 45 graus", rotated)

# Rotaciona a imagem em 90 graus
M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotacionado em 90 graus", rotated)

# Finalmente, vamos usar nossa função auxiliar em utils.py para girar a imagem em 180 graus 
rotated = utils.rotate(image, 180)
cv2.imshow("Rotacionado em 180 graus", rotated)
cv2.waitKey(0)