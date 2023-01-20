# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd
import argparse
from torch.optim import lr_scheduler
from torchvision import transforms
import json
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from neural_net.dataset import SatelliteDataset
from neural_net.image_processor import ProductProcessor
from neural_net.transform import *
from neural_net.index_functions import *
from neural_net.sampler import ShuffleSampler
from neural_net.stopping import EarlyStopping
from neural_net.utils import compute_prec_recall_f1_acc, compute_squared_errors, initialize_weight
from pickle import dump
from sklearn.metrics import confusion_matrix
from collections import OrderedDict, defaultdict
from neural_net import ProductProcessor
from neural_net.unet import UNet
from neural_net.cross_validator import CrossValidator
from neural_net.transform import *
from neural_net.loss import *
from neural_net.performance_storage import *
from neural_net.utils import set_seed
from neural_net.index_functions import nbr
from collections import OrderedDict
from pathlib import Path
from visualize import visualize,normalize
import cv2
def main(args):
    seed = 47
    set_seed(seed)

    epochs = 50
    batch_size = 1
    wd = 0

    validation_dict = {'purple': 'coral',
                   'coral': 'cyan',
                   'pink': 'coral',
                   'grey': 'coral',
                   'cyan': 'coral',
                   'lime': 'coral',
                   'magenta': 'coral'
                  }

    all_bands_selector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    squeeze_mask = False
    print('cuda version detected: %s' % str(torch.version.cuda))
    print('cudnn backend %s' % str(torch.backends.cudnn.version()))

    
    base_result_path = Path('C:\\Users\\mathi\\Documents\\Hackatech\\logs')
    if not base_result_path.is_dir():
        base_result_path.mkdir(parents=True)
    fold_definition = Path('C:\\Users\\mathi\\Documents\\Hackatech\\burned-area-baseline\\satellite_data.csv')

    master_folder = Path('C:\\Users\\mathi\\Documents\\Hackatech\\Satellite_burned_area_dataset')
    csv_path = fold_definition
    n_classes = 2 #len(mask_intervals)
    mask_one_hot = False
    only_burnt = True
    mask_filtering = False
    filter_validity_mask = True
    patience = 5
    tol = 1e-2
    height, width = 512, 512
    mask_postfix='mask'
    groups = OrderedDict()
    df = pd.read_csv(fold_definition)
    grpby = df.groupby('fold')
    for grp in grpby:
        folder_list = grp[1]['folder'].tolist()

        print('______________________________________')
        print('fold key: %s' % grp[0])
        print('folders (%d): %s' % (len(folder_list), str(folder_list)))
        groups[grp[0]] = folder_list

    if not os.path.isdir(base_result_path):
        raise RuntimeError('Invalid base result path %s' % base_result_path)
        
    result_path = base_result_path / 'binary_unet_dice'
    print('##############################################################')
    print('RESULT PATH: %s' % result_path)
    print('##############################################################')
    
    
    lr = 1e-4
    mask_intervals = [(0, 36), (37, 255)]
    product_list = ['sentinel2']
    mode = 'post'
    process_dict = {
        'sentinel2': all_bands_selector,
    }
    n_channels = 12
    ignore_list=None
    transform = transforms.Compose([
        ToTensor(round_mask=True),
        Normalize((0.5, ) * n_channels, (0.5, ) * n_channels)
    ])

    print('#' * 50)
    print('####################### CV all post binary UNET with DiceLoss #######################')
    print('RESULT PATH: %s' % result_path)
    print('BATCH SIZE: %d' % batch_size)
    print('#' * 50)

    model_class = UNet
    model_args = {'n_classes': 2, 'n_channels': n_channels, 'act': 'relu'}
    model = model_class(**model_args)
    device="cuda"
    model.load_state_dict(torch.load(args.model_weights))
    print("model has been loaded")
    validation_set = groups['purple']
    dataset = SatelliteDataset(master_folder, mask_intervals, mask_one_hot, height, width, product_list, mode, filter_validity_mask, transform, process_dict, csv_path, validation_set, ignore_list, mask_filtering, only_burnt, mask_postfix=mask_postfix)
    model.to(device)
   
    print('Dataset dim: %d' % len(dataset))

    
    sampler = ShuffleSampler(dataset, seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
   
    mytype = torch.float32
    for idx, data in enumerate(loader):
        image, mask = data['image'], data['mask']

        image = image.to(device)
        mask = mask.to(device, dtype=mytype)
        if squeeze_mask:
            mask = mask.squeeze(dim=1)

        print(mask.shape)        
        outputs = model(image)
        
        cv2.imshow('net out',torch.logical_and(nn.Softmax(dim=1)(outputs)[0,1]>0.8 , nn.Softmax(dim=1)(outputs)[0,0]<0.1).int().float().cpu().detach().numpy())
        cv2.imshow('real out',(mask[0,0].cpu().detach().numpy()))
      
        input_rgb = visualize(image[0].transpose(0,2).transpose(0,1).cpu().detach().numpy())
        cv2.imshow("input",input_rgb)
        # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
        while  True:
            x = cv2.waitKey(33)
            if(x==ord("a")):
                break
            if(x==ord("q")):
                exit()
        
        # cv2.destroyAllWindows() simply destroys all the windows we created.
        cv2.destroyAllWindows()
        print("ca marche")
        
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights",default="C:\\Users\\mathi\\Documents\\Hackatech\\binary_unet_dice_loss\\magenta_model.pt")
    args = parser.parse_args()
    
    
    main(args)