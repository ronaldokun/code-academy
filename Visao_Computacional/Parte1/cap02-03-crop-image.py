# Crop image

from PIL import Image

# Carrega a imagem que vamos fazer o crop
img = Image.open('images/image01.jpg')

# Cria uma tupla com as dimensões
dim = (100, 100, 400, 400)
crop_img = img.crop(dim)

# Mostra a imagem crop
crop_img.show()
