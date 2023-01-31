# Enhancement da imagem - Muda o Constraste

from PIL import Image
from PIL import ImageEnhance

# Leitura da imagem
img = Image.open('images/image01.jpg')

# Muda o contraste da imagem
enhancer = ImageEnhance.Contrast(img)
new_img = enhancer.enhance(2)

# Mostra a imagem
new_img.show()
