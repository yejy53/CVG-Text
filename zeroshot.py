import warnings
warnings.filterwarnings("ignore")
import os
import sys
from dataset import OSMTextImageDataset
import torch
from torch.utils.data import DataLoader
import argparse
from model_loader import load_model
from utils import find_checkpoint, setup_seed
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP with Distributed Data Parallel")
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--expand', action='store_true')
    parser.add_argument('--img_type', type=str, choices=['sat', 'osm', 'stv'], required=True, 
                        help='Choose the data type: sat, osm, stv')
    parser.add_argument('--checkpoint', type=str, default='')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    setup_seed(42)
    args = parse_args()

    version, model_name, expand, img_type, checkpoint_name = args.version, args.model, args.expand, args.img_type, args.checkpoint

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config['paths'].items():
            if isinstance(value, str) and '{version}' in value:
                value = value.format(version=version)
            globals()[key] = value

    if img_type == 'sat':
        root_dir = sat_root_dir
    elif img_type == 'osm':
        root_dir = osm_root_dir
    elif img_type == 'stv':
        root_dir = stv_root_dir

    if version.endswith("-mixed"):
        testset_path = [testset_path.replace('-mixed', ''), testset_path.replace('-mixed', '-photos')]
        root_dir = [root_dir.replace('-mixed', ''), root_dir.replace('-mixed', '-photos')]

    safe_model_name = model_name.replace('/', '_')
    model_checkpoint_path = os.path.join(checkpoint_dir, safe_model_name)
    
    if checkpoint_name == '':
        checkpoint_name = find_checkpoint(model_checkpoint_path, version, expand, img_type)
    checkpoint_path = os.path.join(model_checkpoint_path, checkpoint_name)

    #output_csv_path = result_dir + '/' + safe_model_name + '/' + f'{checkpoint_name}.csv'
    output_csv_path = os.path.join(result_dir, safe_model_name, f'{checkpoint_name}.csv')
    output_metric_path = os.path.join(result_dir, safe_model_name, f'{checkpoint_name}_metric.csv')

    model, preprocessor, evaluater, forward = load_model(model_name,
                                                checkpoint_path=checkpoint_path,
                                                expand_text=expand,
                                                is_stv=img_type=='stv'
                                                )
    

    dataset = OSMTextImageDataset(root_dir, testset_path, preprocessor=preprocessor)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    print(args)
    print("Test Image Num:", len(dataset))
    print("Image Dir:", root_dir)
    print("Text File:", [testset_path])
    
    os.makedirs(os.path.join(result_dir, safe_model_name), exist_ok=True)
    evaluater(model, dataloader, device, output_csv_path, output_metric_path)