# Desenhando com o OpenCV
 
# Imports
import numpy as np
import cv2

# Inicializando nossa tela como um 300x300 com 3 canais, Vermelho, Verde e Azul, com um fundo preto
canvas = np.zeros((300, 300, 3), dtype = "uint8")

# Desenhando uma linha verde do canto superior esquerdo da nossa tela para o canto inferior direito
green = (0, 255, 0)
cv2.line(canvas, (0, 0), (300, 300), green)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# Agora, desenhamos uma linha vermelha de 3 pixels de espessura do canto superior direito para a parte inferior esquerda
red = (0, 0, 255)
cv2.line(canvas, (300, 0), (0, 300), red, 3)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# Desenhamos um quadrado verde de 50x50 pixels, começando em 10x10 e que termina em 60x60
cv2.rectangle(canvas, (10, 10), (60, 60), green)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# Desenhamos outro retângulo, desta vez vamos torná-lo vermelho e 5 pixels de espessura
cv2.rectangle(canvas, (50, 200), (200, 225), red, 5)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# Vamos desenhar um último retângulo: azul e preenchido
blue = (255, 0, 0)
cv2.rectangle(canvas, (200, 50), (225, 125), blue, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# Reset da nossa tela e desenheando um círculo branco no centro da tela com raios crescentes - de 25 pixels para 150 pixels
canvas = np.zeros((300, 300, 3), dtype = "uint8")
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)

for r in range(0, 175, 25):
	cv2.circle(canvas, (centerX, centerY), r, white)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# Desenhando 25 círculos
for i in range(0, 25):
	# Gerando aleatoriamente um tamanho de raio entre 5 e 200.
	# Gerando uma cor aleatória e, em seguida.
	radius = np.random.randint(5, high = 200)
	color = np.random.randint(0, high = 256, size = (3,)).tolist()
	pt = np.random.randint(0, high = 300, size = (2,))

	# Desenhando o círculo
	cv2.circle(canvas, tuple(pt), radius, color, -1)

# Visualizando o resultado
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)