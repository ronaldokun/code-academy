# Equalização do Histograma

# Para executar este script:
# python cap02-27.1-histogram-equalization.py --image images/estrada.jpg
 
# É possível realizar um cálculo matemático sobre a distribuição de pixels para aumentar o contraste da imagem. 
# A intenção neste caso é distribuir de forma mais uniforme as intensidades dos pixels sobre a imagem. 
# No histograma é possível identificar a diferença pois o acúmulo de pixels próximo a alguns valores é suavizado. 
# Veja a diferença entre o histograma original e o equalizado abaixo.

# Imports
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt 

# Argumento
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho para a imnagem")
args = vars(ap.parse_args())

# Carrega a imagem
image = cv2.imread(args["image"])

# Converte de RGB para Grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# Equalização do Histograma
h_eq = cv2.equalizeHist(image)

# Imagens
cv2.imshow("Original", image)
cv2.imshow("Equalizada", h_eq)
cv2.waitKey(0)

# Histogramas

plt.figure()
plt.title("Histograma Original") 
plt.xlabel("Intensidade") 
plt.ylabel("Total de Pixels") 
plt.hist(image.ravel(), 256, [0,256]) 
plt.xlim([0, 256]) 
plt.show() 

plt.figure() 
plt.title("Histograma Equalizado") 
plt.xlabel("Intensidade") 
plt.ylabel("Total de Pixels") 
plt.hist(h_eq.ravel(), 256, [0,256]) 
plt.xlim([0, 256]) 
plt.show() 

cv2.waitKey(0)






