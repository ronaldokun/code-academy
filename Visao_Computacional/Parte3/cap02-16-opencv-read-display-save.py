# Leitura, Visualização e Gravação de Imagens
 
# Para executar este script:
# python cap02-16-opencv-read-display-save.py --image images/familia.jpg

# Imports
import argparse
import cv2

# Construindo o analisador de argumentos e analisando os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Caminho da imagem")
args = vars(ap.parse_args())

# Carregando a imagem e mostrando algumas informações básicas 
image = cv2.imread(args["image"])
print(f"Altura: {image.shape[0]} pixels")
print(f"Largura: {image.shape[1]} pixels")
print(f"Canais: {image.shape[2]}")

# Mostrando a imagem e aguardando uma tecla pressionada para finalizar
cv2.imshow("Image", image)
cv2.waitKey(0) 

# Salvando uma cópia da imagem
cv2.imwrite("images/nova_familia.jpg", image)