# Convertendo o espaço de cores

from skimage import color, io 

#Read an image from a file
img = io.imread('images/image03.jpg')

#Convert the image tp grayscale
hsv_image = color.rgb2hsv(img)

io.imshow(hsv_image)
io.show()
