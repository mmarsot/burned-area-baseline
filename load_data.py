import pandas as pd
from neural_net.dataset import SatelliteDataset
from torch.utils.data.dataloader import DataLoader
from neural_net.sampler import ShuffleSampler
from neural_net.transform import *
import os
import shutil
#import Path
from zipfile import ZipFile
import argparse

def preprocess_zipfiles(rawdata_folder):
    data_folder = "../data"
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    zip_paths = ["Satellite_burned_area_dataset_part{}.zip".format(i) for i in range(1,6)]
    for zip_path in zip_paths:
        zpath = os.path.join(rawdata_folder, zip_path)
        #if not os.path.exists(zpath):
        print('Extracting %s...' % zpath)
        ZipFile(zpath).extractall(data_folder)

    src_folders = [os.path.join(data_folder,folder, fn) for folder in os.listdir(data_folder) if folder!='.DS_Store' for fn in os.listdir(os.path.join(data_folder,folder))]
    dst_folders = [os.path.join(data_folder, fn) for folder in os.listdir(data_folder) if folder != '.DS_Store'
                   for fn in os.listdir(os.path.join(data_folder, folder))]

    for (src, dst) in zip(src_folders, dst_folders):
        shutil.move(src, dst)

    shutil.copyfile(os.path.join(rawdata_folder, 'satellite_data.csv'), os.path.join(data_folder, 'satellite_data.csv'))
    csv_path = os.path.join(data_folder, 'satellite_data.csv')
    return csv_path

def launch(args):
    if args.data_folder is not None:
        data_folder = args.data_folder
        csv_path = os.path.join(data_folder, 'satellite_data.csv')
    else:
        csv_path = preprocess_zipfiles(args.rawdata_folder)

    df = pd.read_csv(csv_path)
    list_folders = list(df.folder)

    if args.data_path is None:
        master_folder = os.path.join(os.path.split(args.rawdata_folder)[0],"data")
        os.makedirs(master_folder)
    else:
        master_folder = args.data_path

    mask_one_hot = False
    only_burnt = True
    mask_filtering = False
    filter_validity_mask = True
    height, width = 512, 512
    n_channels = 12
    mask_intervals = [(0, 36), (37, 255)]
    all_bands_selector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    product_list = ['sentinel2']
    mode = 'post'
    process_dict = {
        'sentinel2': all_bands_selector,
    }
    test_transform = transforms.Compose([
        ToTensor(round_mask=True),
        Normalize((0.5,) * n_channels, (0.5,) * n_channels)
    ])

    test_dataset = SatelliteDataset(master_folder, mask_intervals, mask_one_hot, height,
                                    width, product_list, mode, filter_validity_mask,
                                    test_transform, process_dict, csv_path, test_set=list_folders,
                                    ignore_list=None, mask_filtering=mask_filtering, only_burnt=only_burnt,
                                    mask_postfix='mask')

    test_sampler = ShuffleSampler(test_dataset, seed = 47)

    batch_size = args.batch_size

    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False)

    return test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument("-data_folder", type=str, default="../data", required=False,
                        help="Path for the folder containing the processed data folder")
    parser.add_argument("-rawdata_folder", type=str, default="../raw_data", required=False,
                        help="Path for the folder containing the raw data folder with the zipfiles")
    parser.add_argument("-batch_size", type=int, default=8, required=False,
                        help="batch size for the test dataloader")

    args = parser.parse_args()
    test_loader = launch(args)
