# Funções para manipulação de imagem
 
# Imports
import numpy as np
import cv2

def translate(image, x, y):
	# Define a matriz de translação e realiza a translação
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Retorna a imagem 
	return shifted

def rotate(image, angle, center = None, scale = 1.0):
	# Obtém as dimensões da imagem
	(h, w) = image.shape[:2]

	# Se o centro for Nenhum, inicialize-o como o centro da imagem
	if center is None:
		center = (w // 2, h // 2)

	# Executa a rotação
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Retorna a imagem 
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# Inicializa as dimensões da imagem a ser redimensionada e obtém o tamanho da imagem
	dim = None
	(h, w) = image.shape[:2]

	# Se tanto a largura quanto a altura são None, então retorna a imagem original
	if width is None and height is None:
		return image

	# Verifica se a largura é Nenhuma
	if width is None:
		# Calcula a proporção da altura e construa as dimensões
		r = height / float(h)
		dim = (int(w * r), height)

	# Caso contrário, a altura é Nenhuma
	else:
		# Calcular a proporção da largura e construir as dimensões
		r = width / float(w)
		dim = (width, int(h * r))

	# Redimensiona a imagem
	resized = cv2.resize(image, dim, interpolation = inter)

	# Retorna a imagem 
	return resized