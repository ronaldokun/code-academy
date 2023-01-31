# Convertendo imagens RGB para escala de cinza

from PIL import Image

# Abre a imagem para leitura
img = Image.open('images/image01.jpg')

# Converte para escala de cinza
gray_image = img.convert("L")

# Mostra a imagem
gray_image.show()
