# Cropping

# Para executar este script:
# python cap02-22-opencv-cropping.py --image images/ferrari.jpg
 
# Imports
import numpy as np
import argparse
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho para imagem")
args = vars(ap.parse_args())

# Carregando a imagem
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Crop de uma imagem é tão simples como usar fatias de matriz em NumPy! 
# Vamos cortar o rosto do menino. 
# A ordem em que especificamos as coordenadas devem ser coletadas: startY: endY, startX: endX
cropped = image[30:190, 260:390]
cv2.imshow("Rosto", cropped)
cv2.waitKey(0)