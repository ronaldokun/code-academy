# Resize

# Para executar este script:
# python cap02-20-opencv-resize.py --image images/familia.jpg
 
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

# Precisamos ter em mente o "aspect ratio" para que a imagem não pareça distorcida.
# Portanto, calculamos a proporção da nova imagem para a imagem antiga. 
# Vamos fazer a nossa nova imagem ter uma largura de 150 pixels
r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r))

# Execute o redimensionamento real da imagem
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Redimensionada (Largura)", resized)

# E se quisermos ajustar a altura da imagem? 
# Aplicamos o mesmo conceito, novamente tendo em mente a relação de aspecto, mas calculando a relação com base na altura.
# Vamos fazer a altura da imagem redimensionada 50 pixels
r = 50.0 / image.shape[0]
dim = (int(image.shape[1] * r), 50)

# Execute o redimensionamento
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Redimensionada (Altura)", resized)
cv2.waitKey(0)

# Claro, calcular a razão cada vez que queremos redimensionar uma imagem é uma dor real. 
# Vamos criar uma função onde podemos especificar nossa largura ou altura alvo, e ter cuidado com o resto para nós.
resized = utils.resize(image, width = 100)
cv2.imshow("Redimensionada via Função", resized)
cv2.waitKey(0)