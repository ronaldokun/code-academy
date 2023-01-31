# Gravando texto na imagem

# Para executar este script:
# python cap02-18.2-opencv-drawing-texto.py --image images/estrada.jpg

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

# Gravando texto na imagem
fonte = cv2.FONT_HERSHEY_SIMPLEX 
cv2.putText(image,'Estrada',(15,65), fonte, 2,(255,255,255),2,cv2.LINE_AA)

# Mostra a imagem
cv2.imshow("Estrada", image) 
cv2.waitKey(0) 