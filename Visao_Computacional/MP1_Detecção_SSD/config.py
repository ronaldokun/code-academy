# Este é o arquivo de configuração
# Aqui estão alguns parâmetros gerais que poderão ser usados por toda aplicação.

class Config:

    # Diretório raiz das imagens VOC
    # O diretório abaixo é do Titan
    VOC_ROOT = '/media/datasets/ComputerVision/Cap02/Mini-Projeto1/dataset/VOC/'

    # Diretório para salvar o modelo (é preciso privilégio de escrita neste diretório)
    # O diretório abaixo é do Titan
    save_folder = 'modelo/'

    # Diretório do modelo pré-treinado
    # O diretório abaixo é do Titan
    basenet = '/media/datasets/ComputerVision/Cap02/Mini-Projeto1/modelo/vgg16_reducedfc.pth'

    # Parâmetro para definir se vamos continuar de onde parou o treinamento
    resume = None

    # Número de classes + 1
    num_classes = 21

    # Número de Épocas
    epoch = 100
    
    # Taxa de aprendizado
    lr = 0.001
    gamma = 0.2
    lr_reduce_epoch = 30

    # Tamanho do batch (lote) de imagens
    batch_size = 32 
    
    # Momentum
    momentum = 0.9

    # Decay
    weight_decay = 5e-4
    
    # Tamanho da imagem de entrada
    min_size = 300
    
    # Tamanho da caixa limitadora
    grids = (38, 19, 10, 5, 3, 1)

    # Número de bounding boxes
    anchor_num = [4, 6, 6, 6, 4, 4]
    
    # 255 * R, G, B
    mean = (104, 117, 123)
    
    # Outros parâmetros
    log_fn = 10 
    neg_radio = 3
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    steps = [s / 300 for s in (8, 16, 32, 64, 100, 300)]
    sizes = [s / 300 for s in (30, 60, 111, 162, 213, 264, 315)] 
    variance = (0.1, 0.2)

opt = Config()
