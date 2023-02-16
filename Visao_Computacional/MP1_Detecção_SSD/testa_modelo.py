# Script para testar o modelo

# Imports
import os
import cv2
import torch
import tqdm
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import opt
from core.utils import nms
from core.voc_dataset import VOCDetection
from core.res_model import RES18_SSD, RES101_SSD
from core.vgg_model import VGG_SSD
from core.multibox_encoder import MultiBoxEncoder
from core.ssd_loss import MultiBoxLoss
from core.augmentations import preproc_for_test
from core.voc_dataset import VOC_LABELS
from core.voc_eval import voc_eval
from core.utils import detect
from core.utils import detection_collate
from collections import OrderedDict

# Verifica os argumentos
parser = argparse.ArgumentParser()

parser.add_argument('--model', 
    default = 'modelo-2685.29.pth',
    type = str,
    help = 'Checkpoint do modelo para testar em novas imagens')

parser.add_argument('--save_folder',
    default = 'output',
    type = str,
    help = 'pasta de output')


args = parser.parse_args()
output_dir = args.save_folder
checkpoint = args.model

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Diretório de imagens de treino
voc_root = opt.VOC_ROOT  

# Join das imagens
annopath = os.path.join(voc_root, 'VOCtest2007', 'VOC2007', 'Annotations', "%s.xml")  
imgpath = os.path.join(voc_root, 'VOCtest2007', 'VOC2007', 'JPEGImages', '%s.jpg')    
imgsetpath = os.path.join(voc_root, 'VOCtest2007', 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')   
cachedir = os.path.join( os.getcwd(), 'annotations_cache')   

# Main
if __name__ == '__main__': 
    
    print('Usando {} para testar o modelo!!'.format(device))
    
    # model = RES18_SSD(opt.num_classes, opt.anchor_num, pretrain=False)
    # model = RES101_SSD(opt.num_classes, opt.anchor_num, pretrain=False)
    # model = VGG_SSD(opt.num_classes, opt.anchor_num)
    model = SSD(opt.num_classes, opt.anchor_num)

    # Carregando o checkpoint
    print('Carregando o checkpoint {}'.format(checkpoint))

    # Estado do dicionário
    state_dict = torch.load(checkpoint, map_location=None if torch.cuda.is_available() else 'cpu')

    # Carrega o modelo
    model.load_state_dict(state_dict)
    model.to(device)
    print('Modelo Carregado')
    
    # Avaliação do modelo
    model.eval()
    multibox_encoder = MultiBoxEncoder(opt)
    
    # Imagens de teste
    image_sets = [['2007', 'test']]
    test_dataset = VOCDetection(opt, image_sets = image_sets, is_train = False)
    
    # Verifica o diretório
    os.makedirs(output_dir, exist_ok = True)
    
    files = [open(os.path.join(output_dir, '{:s}.txt'.format(label)), mode = 'w')
        for label in VOC_LABELS]

    print('Detectando Objetos nas Imagens.........')

    # Loop
    for i in tqdm.tqdm(range(len(test_dataset))):
        src = test_dataset[i][0]
        
        img_name = os.path.basename(test_dataset.ids[i][0]).split('.')[0]
        image = preproc_for_test(src, opt.min_size, opt.mean)
        image = torch.from_numpy(image).to(device)
        with torch.no_grad():
            loc, conf = model(image.unsqueeze(0))
        loc = loc[0]
        conf = conf[0]
        conf = F.softmax(conf, dim=1)
        conf = conf.cpu().numpy()
        loc = loc.cpu().numpy()

        decode_loc = multibox_encoder.decode(loc)
        gt_boxes, gt_confs, gt_labels = detect(decode_loc, conf, nms_threshold = 0.5, gt_threshold = 0.01)

        # Nenhum objecto detectado
        if len(gt_boxes) == 0:
            continue

        h, w = src.shape[:2]
        gt_boxes[:, 0] = gt_boxes[:, 0] * w
        gt_boxes[:, 1] = gt_boxes[:, 1] * h
        gt_boxes[:, 2] = gt_boxes[:, 2] * w
        gt_boxes[:, 3] = gt_boxes[:, 3] * h


        for box, label, score in zip(gt_boxes, gt_labels, gt_confs):
            print(img_name, "{:.3f}".format(score), "{:.1f} {:.1f} {:.1f} {:.1f}".format(*box), file = files[label])


    for f in files:
        f.close()
    
    
    print('Calcula MAP.........')
    aps = []
    for f in os.listdir(output_dir):
        filename = os.path.join(output_dir, f)
        class_name = f.split('.')[0]
        rec, prec, ap = voc_eval(filename, annopath, imgsetpath.format('test'), class_name, cachedir, ovthresh = 0.1, use_07_metric = True)
        print(class_name, ap)
        aps.append(ap)

    print('MAP Médio: ', np.mean(aps))
    




