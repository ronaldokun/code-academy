# Convertendo o espaço de cor

from skimage import color, io 

# Leitura
img = io.imread('images/image02.jpg')

# Convertendo de RGB para Grayscale
gray_image = color.rgb2gray(img)

io.imshow(gray_image)
io.show()
