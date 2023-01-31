# Transformação Geométrica

from PIL import Image

# Leitura da Imagem
img = Image.open('images/image01.jpg')

# Rotaciona 90 graus
rotated_img = img.rotate(90)

# Mostra a imagem
rotated_img.show()
