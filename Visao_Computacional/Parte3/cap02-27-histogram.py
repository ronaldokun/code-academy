# Histograma

# Para executar este script:
# python cap02-27-histogram.py --image images/estrada.jpg
 
# Um histograma é um gráfico de colunas ou de linhas que representa a distribuição dos valores dos pixels de uma imagem, 
# ou seja, a quantidade de pixeis mais claros (próximos de 255) e a quantidade de pixels mais escuros (próximos de 0).
# O eixo X do gráfico normalmente possui uma distribuição de 0 a 255 que demonstra o valor (intensidade) do pixel e 
# no eixo Y é plotada a quantidade de pixels daquela intensidade.

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

# Calcula o histograma
hist = cv2.calcHist([image], [0], None, [256], [0, 256]) 

# Histograma
plt.figure() 
plt.title("Histograma em Tons de Cinza") 
plt.xlabel("Intensidade") 
plt.ylabel("Total de Pixels") 
plt.plot(hist) 
plt.xlim([0, 256]) 
plt.show() 
cv2.waitKey(0)

# Também é possível plotar o histograma de outra forma, com a ajuda da função "ravel()". 
# Neste caso o eixo X avança o valor 255 indo até 300, espaço que não existem pixels.
plt.hist(image.ravel(),256,[0,256]) 
plt.show()

# Além do histograma da imagem em tons de cinza é possível plotar um histograma da imagem colorida. 
# Neste caso teremos três linhas, uma para cada canal. 
# Note que a função "zip" cria uma lista de tuplas formada pelas união das listas passadas 
#Separa os canais
image = cv2.imread('images/estrada.jpg')
canais = cv2.split(image) 
cores = ("b", "g", "r") 

plt.figure() 
plt.title("'Histograma Colorido") 
plt.xlabel("Intensidade") 
plt.ylabel("Total de Pixels") 
for (canal, cor) in zip(canais, cores):
	# Este loop executa 3 vezes, uma para cada canal 
	hist = cv2.calcHist([canal], [0], None, [256], [0, 256]) 
	plt.plot(hist, color = cor) 
	plt.xlim([0, 256]) 
plt.show()

