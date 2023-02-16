# Métodos de Interpolação

import cv2
import imageio
import itertools
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt 
import pandas as pd
import time

# Imagens
ref_images = ["cafe.jpg", "cavalo.jpg"] 

# Limita o tamanho original
images_orig = [cv2.resize(imageio.imread(im), (400,400)) for im in ref_images] 

# Métodos de Interpolação
methods=[("area", cv2.INTER_AREA), 
         ("nearest", cv2.INTER_NEAREST), 
         ("linear", cv2.INTER_LINEAR), 
         ("cubic", cv2.INTER_CUBIC), 
         ("lanczos4", cv2.INTER_LANCZOS4)]

# Versão do OpenCV
print ("opencv version", cv2.__version__)

# Função para mostrar as imagens
def display(images, titles=['']):
    if isinstance(images[0], list):
        c = len(images[0])
        r = len(images)
        images = list(itertools.chain(*images))
    else:
        c = len(images)
        r = 1
    plt.figure(figsize=(4*c, 4*r))
    gs1 = gridspec.GridSpec(r, c, wspace=0, hspace=0)
    titles = itertools.cycle(titles)
    for i in range(r*c):
        im = images[i]
        title = next(titles)
        plt.subplot(gs1[i])
        plt.imshow(im, cmap='gray', interpolation='none')
        plt.axis('off')
        if i < c:
            plt.title(title)
    plt.tight_layout()


# Interpolação
images_small = [cv2.resize(im, (50,50),interpolation=cv2.INTER_AREA) for im in images_orig]
image_set = [[cv2.resize(im, (400,400), interpolation=m[1]) for m in methods] for im in images_small]
image_set = [[ima,]+imb for ima, imb in zip(images_small, image_set)]
names = ["original 50x50",] + [m[0]+" 400x400" for m in methods]
display(image_set, names)
plt.savefig("opencv_interpolation.jpg", dpi=75)


