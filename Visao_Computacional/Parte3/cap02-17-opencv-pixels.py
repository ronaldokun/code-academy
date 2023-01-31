# Manipulando pixels

# Para executar este script:
# python cap02-17-opencv-pixels.py --image images/familia.jpg
 
# Imports
import argparse
import cv2

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho da imagem")
args = vars(ap.parse_args())

# Carregando a imagem
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Imagens são simplesmente arrays NumPy. Um pixel no canto superior esquerdo pode ser encontrado como (0, 0)
(b, g, r) = image[0, 0]
print("Pixel em (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

# É importante notar que o OpenCV armazena canais RGB na ordem inversa. 
# Embora normalmente pensemos em termos de Vermelho, Verde e Azul, o OpenCV os armazena na ordem de Azul, Verde e Vermelho.
# Mas por que ele faz isso, armazena imagens no BGR em vez da ordem RGB?
# A resposta é eficiência. As imagens multicanal no OpenCV são armazenadas na ordem das linhas,  
# o que significa que cada um dos componentes Azul, Verde e Vermelho são concatenados em sub-colunas para formar 
# a imagem inteira. Além disso, cada linha da matriz deve ser alinhada a um limite de 4 bytes, 
# um byte para cada um dos canais vermelho, verde, azul e alfa. 
# Dada nossa imagem, a última linha da imagem vem primeiro na memória, assim nós armazenamos os componentes na ordem inversa.

# Agora mudamos o valor do pixel
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print("Pixel em (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

# Como estamos usando matrizes numPy, podemos aplicar slicing e obter grandes pedaços da imagem. 
# Vamos pegar a parte superior esquerda
corner = image[0:100, 0:100]
cv2.imshow("Corner", corner)

# Mudamos o canto superior esquerdo para verde
image[0:100, 0:100] = (0, 255, 0)

# Mostra a imagem
cv2.imshow("Updated", image)
cv2.waitKey(0)