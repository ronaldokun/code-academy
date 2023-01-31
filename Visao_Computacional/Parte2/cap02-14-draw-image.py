# Desenhando um círculo

import numpy as np 
from skimage import io, draw 

# Leitura
img = np.zeros((100, 100), dtype=np.uint8)

# Desenha um círculo
x, y = draw.circle(50, 50, 10)
img[x, y] = 1

io.imshow(img)
io.show()
