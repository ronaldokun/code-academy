# Transformação Geométrica

from PIL import Image

# Leitura da imagem
img = Image.open('images/image01.jpg')

# Resize
resize_img = img.resize((300, 200))

# Mostra a imagem
resize_img.show()
