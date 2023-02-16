# Script para o banco de dados de imagens

# Imports
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import torch.utils.data as data
import xml.etree.ElementTree as ET
from config import opt
from core.augmentations import preproc_for_test, preproc_for_train

# Labels das classes
VOC_LABELS = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',)

# Classe para criar o banco de imagens de treino
class VOCDetection(data.Dataset):

    # Construtor
    def __init__(self, opt, image_sets = [['2007', 'trainval'], ['2012', 'trainval']], is_train = True):

        # Inicializa atributos
        self.root = opt.VOC_ROOT  
        self.image_sets = image_sets
        self.is_train = is_train
        self.opt = opt   
        self.ids = []
        
        # Carrega as imagens
        for (year, name) in self.image_sets:
            root_path = os.path.join(self.root, 'VOC' + name + year)
            root_path = os.path.join(root_path, 'VOC' + year)
            ano_file = os.path.join(root_path, 'ImageSets', 'Main', name + '.txt')
    
            # Busca imagens e labels (annotations)
            with open(ano_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    ano_path = os.path.join(root_path, 'Annotations', line + '.xml')
                    img_path = os.path.join(root_path, 'JPEGImages', line + '.jpg')
                    self.ids.append((img_path, ano_path))

    # Método para obter uma imagem
    def __getitem__(self, index):
        img_path, ano_path = self.ids[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        boxes, labels = self.get_annotations(ano_path)
        
        # Processa as imagens para treinamento
        if self.is_train:
            image, boxes, labels = preproc_for_train(image, boxes, labels, opt.min_size, opt.mean)
            image = torch.from_numpy(image)
           
        # Concatena imagem e caixa delimitadora (box)
        target = np.concatenate([boxes, labels.reshape(-1,1)], axis = 1)
        
        return image, target

    # Método para obter as anotações (labels)
    def get_annotations(self, path):
        
        tree = ET.parse(path)

        boxes = []
        labels = []
        
        for child in tree.getroot():
            if child.tag != 'object':
                continue

            bndbox = child.find('bndbox')

            box = [float(bndbox.find(t).text) - 1 for t in ['xmin', 'ymin', 'xmax', 'ymax']]

            label = VOC_LABELS.index(child.find('name').text) 
            
            boxes.append(box)

            labels.append(label)


        return np.array(boxes), np.array(labels)

    
    def __len__(self):
        return len(self.ids)
    


