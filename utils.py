import torch
from torch import nn
import numpy as np
import random
from torch.nn import functional as F
import os, sys, logging
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import re

class PrintLogger(object):
    def __init__(self, logger):
        self.logger = logger
        self.terminal = sys.stdout

    def write(self, message):
        if message.strip() != "":
            self.logger.info(message.strip())

    def flush(self):
        pass

def setup_logging(logdir, safe_model_name, hyperparams):
    
    log_dir = os.path.join(logdir, safe_model_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{hyperparams}.log')

    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ])
    sys.stdout = PrintLogger(logging.getLogger())
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     #torch.backends.cudnn.deterministic = True

def interpolate_pos_embedding(pos_embedding, new_seq_len):
    from torch.nn import functional as F
    pos_embedding = pos_embedding.unsqueeze(0).permute(0, 2, 1)
    pos_embedding = F.interpolate(pos_embedding, size=new_seq_len, mode='linear', align_corners=False)
    pos_embedding = pos_embedding.permute(0, 2, 1).squeeze(0) 
    return pos_embedding
        
def resize_image(processor, model):
    for t in processor.transforms:
        if isinstance(t, Resize) or isinstance(t, CenterCrop):
            if isinstance(t.size, int):
                original_size = t.size
            elif isinstance(t.size, tuple):
                original_size = t.size[0]
            t.size = (original_size, original_size * 2)
    print(original_size)
    return (original_size, original_size * 2)

def resize_img_pos_emb(model, image_height, image_width, rank=0):

    def get_patch_size(model):
        if hasattr(model.visual, 'patch_embed') and hasattr(model.visual.patch_embed, 'patch_size'):
            patch_size = model.visual.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            return patch_size
        elif hasattr(model.visual, 'conv1') and hasattr(model.visual.conv1, 'stride'):
            patch_size = model.visual.conv1.stride[0]
            return patch_size
        else:
            raise ValueError("Patch size not found in the model's visual encoder")

    patch_size = get_patch_size(model)

    new_height = image_height // patch_size
    new_width = image_width // patch_size

    original_pos_embedding = model.visual.positional_embedding
    device = original_pos_embedding.device

    cls_token = original_pos_embedding[:1, :] 
    spatial_pos_embedding = original_pos_embedding[1:, :]

    original_grid_height = new_height 
    original_grid_width = spatial_pos_embedding.size(0) // original_grid_height

    if original_grid_height == new_height and original_grid_width == new_width:
        if rank == 0:
            print(f"Positional embedding matched: {original_grid_height}x{original_grid_width}.")
        return

    spatial_pos_embedding = spatial_pos_embedding.view(1, original_grid_height, original_grid_width, -1)

    new_spatial_pos_embedding = F.interpolate(
        spatial_pos_embedding.permute(0, 3, 1, 2),
        size=(new_height, new_width), 
        mode='bicubic', 
        align_corners=False
    ).permute(0, 2, 3, 1).view(new_height * new_width, -1)


    new_pos_embedding = torch.cat([cls_token, new_spatial_pos_embedding], dim=0)
    model.visual.positional_embedding = torch.nn.Parameter(new_pos_embedding).to(device)

    if rank == 0:
        print(f"Positional embedding resized from {original_grid_height}x{original_grid_width} to {new_height}x{new_width}.")

    
def resize_img_pos_emb_siglip(model, image_height, image_width, rank=0):

    def get_patch_size(model):
        if hasattr(model.vision_model.embeddings, 'patch_embedding') and hasattr(model.vision_model.embeddings.patch_embedding, 'kernel_size'):
            patch_size = model.vision_model.embeddings.patch_embedding.kernel_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            return patch_size
        else:
            raise ValueError("Patch size not found in the model's vision embeddings.")
    
    patch_size = get_patch_size(model)
    new_height = image_height // patch_size
    new_width = image_width // patch_size

    original_pos_embedding = model.vision_model.embeddings.position_embedding.weight
    device = original_pos_embedding.device

    cls_token = original_pos_embedding[:1, :] 
    spatial_pos_embedding = original_pos_embedding[1:, :] 

    original_num_patches = spatial_pos_embedding.size(0)
    original_grid_height = int((original_num_patches) ** 0.5)
    original_grid_width = original_num_patches // original_grid_height

    if original_grid_height == new_height and original_grid_width == new_width:
        if rank == 0:
            print(f"Positional embedding matched: {original_grid_height}x{original_grid_width}.")
        return


    spatial_pos_embedding = spatial_pos_embedding.view(1, original_grid_height, original_grid_width, -1)

    new_spatial_pos_embedding = F.interpolate(
        spatial_pos_embedding.permute(0, 3, 1, 2), 
        size=(new_height, new_width), 
        mode='bicubic', 
        align_corners=False
    ).permute(0, 2, 3, 1).view(new_height * new_width, -1)

    new_pos_embedding = torch.cat([cls_token, new_spatial_pos_embedding], dim=0)
    model.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(new_pos_embedding).to(device)

    if rank == 0:
        print(f"Positional embedding resized from {original_grid_height}x{original_grid_width} to {new_height}x{new_width}.")


def find_checkpoint(checkpoint_dir, version, expand, img_type):
    if expand:
        pattern = re.compile(rf"long_model_{version}_.*{img_type}.*\.pth")
    else:
        pattern = re.compile(rf"model_{version}_.*{img_type}.*\.pth")

    all_checkpoints = os.listdir(checkpoint_dir)
    matching_checkpoints = [ckpt for ckpt in all_checkpoints if pattern.match(ckpt)]
    
    if len(matching_checkpoints) == 0:
        print(f"No checkpoint found for version={version}, expand={expand} and img_type={img_type}")
        matching_checkpoints = ['model.pt']
    matching_checkpoints.sort()
    return matching_checkpoints[0]

def sanitize_filename(filename):

    return filename.replace(',', '_')