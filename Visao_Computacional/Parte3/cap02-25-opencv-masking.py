# Masking

# Para executar este script:
# python cap02-25-opencv-masking.py --image images/familia.jpg
 
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

# O mascaramento nos permite focar apenas em partes de uma imagem que
# nos interessa. Uma máscara é do mesmo tamanho que a nossa imagem, mas tem
# apenas dois valores de pixel, 0 e 255. Pixels com um valor de 0
# são ignorados na imagem orignal e pixels com um valor de 255 podem ser mantidos. 
# Por exemplo, vamos construir uma máscara com um quadrado de 150x150 no centro da imagem.
mask = np.zeros(image.shape[:2], dtype = "uint8")
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75 , cY + 75), 255, -1)
cv2.imshow("Mask", mask)

# Aplique a máscara - observe como apenas a região retangular central é mostrada
masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Mascara aplicada", masked)
cv2.waitKey(0)

# Agora, vamos fazer uma máscara circular com um raio de 100 pixels
mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.circle(mask, (cX, cY), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Mask", mask)
cv2.imshow("Mascara aplicada", masked)
cv2.waitKey(0)