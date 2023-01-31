# Enhancement da imagem - Muda o Brilho

from PIL import Image
from PIL import ImageEnhance

# Leitura da imagem
img = Image.open('images/image01.jpg')

# Enhancement
enhancer = ImageEnhance.Brightness(img)
bright_img = enhancer.enhance(2)

# Mostra a imagem
bright_img.show()
