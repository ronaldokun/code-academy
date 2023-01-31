# Flipping

# Para executar este script:
# python cap02-21-opencv-flipping.py --image images/ferrari.jpg

# Imports
import argparse
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho para a imagem")
args = vars(ap.parse_args())

# Carregando os dados
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Flip horizontal
flipped = cv2.flip(image, 1)
cv2.imshow("Flip horizontal", flipped)

# Flip vertical
flipped = cv2.flip(image, 0)
cv2.imshow("Flip vertical", flipped)

# Flip horizontal e vertical
flipped = cv2.flip(image, -1)
cv2.imshow("Flip horizontal e vertical", flipped)
cv2.waitKey(0)