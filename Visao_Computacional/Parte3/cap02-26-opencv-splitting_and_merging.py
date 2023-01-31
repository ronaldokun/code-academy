# Splitting e Merging

# Para executar este script:
# python cap02-26-opencv-splitting_and_merging.py --image images/estrada.jpg
 
# Imports
import numpy as np
import argparse
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho para a imagem")
args = vars(ap.parse_args())

# Carrega a imagem e obtém cada canal: Vermelho, Verde e Azul
# NOTA: OpenCV armazena uma imagem como NumPy array com os
# canais na ordem inversa! Quando chamamos cv2.split, estamos
# realmente obtendo os canais como Azul, Verde, Vermelho!
image = cv2.imread(args["image"])
(B, G, R) = cv2.split(image)

# Agora, vamos visualizar cada canal em cores
zeros = np.zeros(image.shape[:2], dtype = "uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)

# Junta novamente a imagem 
merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()