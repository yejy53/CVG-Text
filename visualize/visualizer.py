import warnings
warnings.filterwarnings("ignore")
import os
import sys
import torch
from torch.utils.data import DataLoader
from model_loader import load_model

from dataset import OSMTextImageDataset
from collections import defaultdict
import argparse
from utils import find_checkpoint
import yaml
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP with Distributed Data Parallel")
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--expand', action='store_true')
    parser.add_argument('--img_type', type=str, choices=['sat', 'osm', 'stv'], required=True, 
                        help='Choose the data type: sat, osm, stv')
    parser.add_argument('--osm_text', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='')

    args = parser.parse_args()
    return args

def collect_attention_statistics(model, csv_file_path, source_dir, output_path, max_samples=1000, min_frequency=5):
    import json, csv, pandas as pd
    from PIL import Image
    from visualize.interpreter import interpret, show_image_relevance, show_text_relevance
    import numpy as np
    import shutil

    word_attention_scores = defaultdict(list)
    df = pd.read_csv(csv_file_path)
    
    for idx, row in tqdm(df.iterrows()):
        if idx >= max_samples:
            break
        search_query = row['text'] 
        origin_img = row['original_image']
        top_5_images = json.loads(row['top_5_matches'])
        top_5_images.append(origin_img)
        top_image_paths = []
        top_images = []
        for i, top_image in enumerate(top_5_images):
            top_image_path = os.path.join(source_dir, top_image)
            if not os.path.exists(top_image_path):
                print(f"Top-{i} candidate image not found: {top_image_path}")
                continue
            top_image_paths.append(top_image)
            top_image = Image.open(top_image_path).convert("RGB")
            top_images.append(top_image)

        img, text = dataloader.dataset.preprocessor(top_images, [search_query]*6)
        img = img.to(device)
        text = text.to(device)
        R_text, R_image = interpret(img, text, model, device)

        case_dir = os.path.join(output_path, origin_img)
        os.makedirs(case_dir, exist_ok=True)
        shutil.copy(os.path.join(source_dir, origin_img), os.path.join(case_dir, origin_img))

        top1_text_save_path = os.path.join(case_dir, 'top1_text_attn.png')
        origin_text_save_path = os.path.join(case_dir, 'text_attn.png')
        show_text_relevance(search_query, text[0], R_text[0],top1_text_save_path)
        show_text_relevance(search_query, text[5], R_text[5],origin_text_save_path)
        for i, image_relevance in enumerate(R_image[:-1]):
            image_save_path = os.path.join(case_dir, f'top{i}_img_attn_{top_image_paths[i]}')
            show_image_relevance(image_relevance, img[i], top_images[i], image_save_path)


if __name__ == "__main__":
    args = parse_args()

    version, model_name, expand, img_type, osm, checkpoint_path = args.version, args.model, args.expand, args.img_type, args.osm_text, args.checkpoint

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config['paths'].items():
            if isinstance(value, str) and '{version}' in value:
                value = value.format(version=version)
            globals()[key] = value


    safe_model_name = model_name.replace('/', '_')
    if checkpoint_path == '':
        model_checkpoint_path = os.path.join(checkpoint_dir, safe_model_name)

        checkpoint_name = find_checkpoint(model_checkpoint_path, version, expand, img_type)
        checkpoint_path = os.path.join(model_checkpoint_path, checkpoint_name)

    else:
        checkpoint_name = os.path.basename(checkpoint_path)

    csv_file_path = os.path.join(result_dir, safe_model_name, f'{checkpoint_name}.csv')
    model, preprocessor, evaluater, forward = load_model(model_name,
                                                checkpoint_path=checkpoint_path,
                                                expand_text=expand,
                                                )
    
    if img_type == 'sat':
        root_dir = sat_root_dir
    elif img_type == 'osm':
        root_dir = osm_root_dir
    elif img_type == 'stv':
        root_dir = stv_root_dir

    dataset = OSMTextImageDataset(root_dir, testset_path, preprocessor=preprocessor)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)

    output_path = os.path.join(case_dir, f'{version}_{img_type}', safe_model_name)
    os.makedirs(output_path, exist_ok=True)

    model.to(device)
    model.eval()

    collect_attention_statistics(model, csv_file_path, root_dir, output_path, max_samples=1, min_frequency=0)