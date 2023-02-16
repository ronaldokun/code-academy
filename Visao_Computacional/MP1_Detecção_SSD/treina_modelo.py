# Script para treinar o modelo

# Imports
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import opt
from core.res_model import RES18_SSD, RES101_SSD
from core.vgg_model import VGG_SSD
from core.ssd import SSD
from core.resnet import *
from core.utils import detection_collate
from core.multibox_encoder import MultiBoxEncoder
from core.ssd_loss import MultiBoxLoss
from core.voc_dataset import VOCDetection

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Função para ajustar a taxa de aprendizado durante o treinamento 
# Define a redução da taxa de aprendizado a cada 10 steps
def adjust_learning_rate(optimizer, gamma, step):
    lr = opt.lr * (gamma ** (step))
    print('Alterando a taxa de aprendizado para:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Main
if __name__ == '__main__':

    print('\nExecutando no Device : ', device)

    # Se a pasta de saída não existir, será criada
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
     
    # Cria instância do modelo
    model = SSD(opt.num_classes, opt.anchor_num)
    
    # Carrega o checkpoint ou começa o treinamento do zero
    if opt.resume:
        print('Carregando Checkpoint...')
        model.load_state_dict(torch.load(opt.resume))
    else:
        vgg_weights = torch.load(opt.basenet)
        print('Carregando a Rede Pré-Treinada...')
        model.vgg.load_state_dict(vgg_weights)

    # Envia o modelo para o device     
    model.to(device)

    # Treina o modelo
    model.train()

    # Encoder das caixas delimitadoras
    mb = MultiBoxEncoder(opt)
        
    # Datasets de treino
    image_sets = [['2007', 'trainval'], ['2012', 'trainval']]

    # Prepara os dados
    dataset = VOCDetection(opt, image_sets = image_sets, is_train = True)

    # Carrega os dados
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, collate_fn = detection_collate, num_workers = 4)

    # Calcula o erro
    criterion = MultiBoxLoss(opt.num_classes, opt.neg_radio).to(device)

    # Otimizador
    optimizer = torch.optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)

    print('Iniciando o Treinamento........')

    for e in range(opt.epoch):

        # Define se vamos alterar a taxa de aprendizado neste step
        if e % opt.lr_reduce_epoch == 0:
            adjust_learning_rate(optimizer, opt.gamma, e//opt.lr_reduce_epoch)
        
        # Inicializa totais
        total_loc_loss = 0
        total_cls_loss = 0
        total_loss = 0

        # Loop
        for i , (img, boxes) in enumerate(dataloader):
            img = img.to(device)
            gt_boxes = []
            gt_labels = []
            for box in boxes:
                labels = box[:, 4]
                box = box[:, :-1]
                match_loc, match_label = mb.encode(box, labels)
                gt_boxes.append(match_loc)
                gt_labels.append(match_label)
            
            gt_boxes = torch.FloatTensor(gt_boxes).to(device)
            gt_labels = torch.LongTensor(gt_labels).to(device)
            p_loc, p_label = model(img)
            loc_loss, cls_loss = criterion(p_loc, p_label, gt_boxes, gt_labels)
            loss = loc_loss + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()
            
            if i % opt.log_fn == 0:
                avg_loc = total_loc_loss / (i+1)
                avg_cls = total_cls_loss / (i+1)
                avg_loss = total_loss / (i+1)
                print('Epoch[{}] | batch_idx[{}] | loc_loss [{:.2f}] | cls_loss [{:.2f}] | total_loss [{:.2f}]'.format(e, i, avg_loc, avg_cls, avg_loss))
                
        # Salva o modelo periodicamente
        if (e+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_folder, 'modelo-{:.2f}.pth'.format(total_loss)))

# Salva a última versão do modelo
torch.save(model.state_dict(), os.path.join(opt.save_folder, 'modelo-final.pth'.format(total_loss)))


