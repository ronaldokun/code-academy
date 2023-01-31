# Leitura da imagem com skimage

# pip install scikit-image

from skimage import io 

img = io.imread('images/image02.jpg')
io.imshow(img)
io.show()

