# Salvando a imagem

from skimage import io 

# Leitura da imagem
img = io.imread('images/image02.jpg')

# Salvando como um novo arquivo
io.imsave('images/new_image.jpg', img)
