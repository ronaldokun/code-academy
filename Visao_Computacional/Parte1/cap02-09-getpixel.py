# GetPixel

from PIL import Image

img = Image.open('images/image01.jpg')

print(img.getpixel((100,100)))
