import torch
from torchvision import transforms
import sys
import os
import yaml
import clip
from transformers import AutoProcessor, AutoModel, BlipForImageTextRetrieval, ViltForImageAndTextRetrieval, ViltProcessor

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

os.environ["HF_ENDPOINT"] = config["hf_endpoint"]
sys.path.extend(config["custom_paths"])

from utils import resize_image, resize_img_pos_emb, resize_img_pos_emb_siglip, interpolate_pos_embedding
from evaluate import evaluate_clip, evaluate_siglip, evaluate_blip, evaluate_vilt, evaluate_clip_osm
from trainer import clip_forward, siglip_forward, blip_forward, clip_forward_osm
#import open_clip
#from eva_clip import create_model_and_transforms, get_tokenizer

MAX_TEXT_LEN = 300
MODEL_LIST = ["CLIP-B/16", "CLIP-L/14@336",
              "OpenCLIP-L/14", "OpenCLIP-H/14"
              "CLIPA-L/14@336", "CLIPA-H/14@336"
              "EVA2-CLIP-B/16", "EVA2-CLIP-L/14@336",
              "SigLIP-B/16", "SigLIP-L/16@384", "SigLIP-SO400M", 
              "BLIP-B", "BLIP-L",
              "ViLT"]

def load_model(model_name, expand_text=False, checkpoint_path='', is_stv=False):
    pretrained_paths = config["pretrained_paths"]

    if model_name.lower() == 'clip-b/16':
        model_path = 'ViT-B/16'
        model, preprocess = clip.load(model_path, device='cpu')
        model = model.float()
        print("Text Pos Emb:", model.positional_embedding.shape)
 
    elif model_name.lower() == 'clip-l/14@336':
        model_path = 'ViT-L/14@336px'
        model, preprocess = clip.load(model_path, device='cpu')
        model = model.float()
        print("Text Pos Emb:", model.positional_embedding.shape)
    
    elif model_name.lower() == 'openclip-l/14':
        model_path = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=pretrained_paths["openclip_l14"])
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

        model = model.float()
        print("Text Pos Emb:", model.positional_embedding.shape)
        
    elif model_name.lower() == 'openclip-h/14':
        model_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained=pretrained_paths["openclip_h14"])
        tokenizer = open_clip.get_tokenizer('ViT-H-14')

        model = model.float()
        print("Text Pos Emb:", model.positional_embedding.shape)

    elif model_name.lower() == 'eva2-clip-b/16':
        model_path = "EVA02-CLIP-B-16"
        model, _, preprocess = create_model_and_transforms(model_path, pretrained_paths["eva2_clip_b16"], force_custom_clip=True)
        tokenizer = get_tokenizer(model_path)

        model = model.float()
        print("Text Pos Emb:", model.text.positional_embedding.shape)

    elif model_name.lower() == 'eva2-clip-l/14@336':
        model_path = "EVA02-CLIP-L-14-336"
        model, _, preprocess = create_model_and_transforms(model_path, pretrained_paths["eva2_clip_l14_336"], force_custom_clip=True)
        tokenizer = get_tokenizer(model_path)

        model = model.float()
        print("Text Pos Emb:", model.text.positional_embedding.shape)

    elif model_name.lower() == 'siglip-b/16':
        model_path = "google/siglip-base-patch16-256"
        model = AutoModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)  

    elif model_name.lower() == 'siglip-l/16@384':
        model_path = "google/siglip-large-patch16-384"
        model = AutoModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        
    elif model_name.lower() == 'siglip-so400m':
        model_path = "google/siglip-so400m-patch14-384"
        model = AutoModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        
    elif model_name.lower() == 'blip-b':
        model_path = pretrained_paths["blip_b"]
        model = BlipForImageTextRetrieval.from_pretrained(model_path,local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_path,local_files_only=True)

    elif model_name.lower() == 'blip-l':
        model_path = pretrained_paths["blip_l"]
        model = BlipForImageTextRetrieval.from_pretrained(model_path,local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_path,local_files_only=True)

    elif model_name.lower() == 'vilt':
        model_path = 'dandelin/vilt-b32-finetuned-coco'
        model = ViltForImageAndTextRetrieval.from_pretrained(model_path)
        processor = ViltProcessor.from_pretrained(model_path)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    if model_name.lower() in ['clip-b/16', 'clip-l/14@336']:
        
        if expand_text:
            model.positional_embedding = torch.nn.Parameter(interpolate_pos_embedding(model.positional_embedding, MAX_TEXT_LEN))
            mask = torch.empty(300, 300)
            mask.fill_(float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal
            for block in model.transformer.resblocks:
                block.attn_mask = mask
            print("New Position Embedding Shape:", model.positional_embedding.shape) 
        
        if is_stv:
            img_size = resize_image(preprocess, model)
            print('IMG SIZE:', img_size)
            
            resize_img_pos_emb(model, img_size[0], img_size[1])

        def preprocessor(image,text):
            if isinstance(image, list):
                image = torch.stack([preprocess(i) for i in image])
            else:
                image = preprocess(image)
            if expand_text:
                text_id = clip.tokenize(text, context_length=MAX_TEXT_LEN,truncate=True).squeeze(0)
            else:
                text_id = clip.tokenize(text, truncate=True).squeeze(0)
            return image, text_id
        
        evaluater = evaluate_clip
        forward = clip_forward

    elif 'siglip' in model_name.lower():
        if expand_text:
            pos_embedding = torch.nn.Parameter(interpolate_pos_embedding(model.text_model.embeddings.position_embedding.weight, MAX_TEXT_LEN))
            model.text_model.embeddings.register_buffer(
                "position_ids", torch.arange(MAX_TEXT_LEN).expand((1, -1)), persistent=False
            )
            model.text_model.embeddings.position_embedding.weight = pos_embedding
            print("New Position Embedding Shape:", model.text_model.embeddings.position_embedding.weight.shape) 

        if is_stv:
            processor.image_processor.size['width'] = processor.image_processor.size['height'] * 2
            print('IMG SIZE:', (processor.image_processor.size['height'], processor.image_processor.size['width']))
            raise NotImplementedError('resize_img_pos_emb_siglip')
            resize_img_pos_emb_siglip(model, processor.image_processor.size['height'], processor.image_processor.size['width'])
            

        def preprocessor(image,text):
            if expand_text:
                output = processor(text=[text], images=image, padding='max_length', return_tensors="pt", truncation=True, max_length=MAX_TEXT_LEN)
            else:
                output = processor(text=[text], images=image, padding='max_length', return_tensors="pt", truncation=True, max_length=64)
            return output['pixel_values'].squeeze(0), output['input_ids']
        
        evaluater = evaluate_siglip
        forward = siglip_forward
    
    elif 'openclip' in model_name.lower() or 'eva' in model_name.lower():
        if expand_text:
            resize_text_pos_emb_openclip(model)

        if is_stv:
            img_size = resize_image(preprocess, model)
            print('IMG SIZE:', img_size)
            
            resize_img_pos_emb(model, img_size[0], img_size[1])

        def preprocessor(image,text):
            image = preprocess(image)
            if expand_text:
                text_id = tokenizer(text, context_length=MAX_TEXT_LEN).squeeze(0)
            else:
                text_id = tokenizer(text).squeeze(0)
            return image, text_id
        
        evaluater = evaluate_clip
        forward = clip_forward

    elif 'blip' in model_name.lower():
        if is_stv:
            raise NotImplementedError
        def preprocessor(image,text):
            output = processor(text=[text], images=image, padding='max_length', return_tensors="pt", truncation=True, max_length=MAX_TEXT_LEN)
            return output['pixel_values'].squeeze(0), output['input_ids'].squeeze(0)
        
        evaluater = evaluate_blip
        forward = blip_forward

    elif 'vilt' in model_name.lower():

        preprocessor = processor
        
        evaluater = evaluate_vilt
        forward = siglip_forward

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if os.path.exists(checkpoint_path):
        print(f"Loading model parameters from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location = 'cpu')
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Using default model parameters.")

    return model, preprocessor, evaluater, forward
