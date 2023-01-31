# Salvando uma nova imagem

from PIL import Image

# Abre a imagem
img = Image.open('images/image01.jpg')

# Salva com nome diferente
img.save('images/new_image.jpg')

