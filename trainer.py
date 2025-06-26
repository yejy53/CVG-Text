import torch
from torch import nn
import torch.distributed as dist

def gather_tensor(tensor):

    world_size = dist.get_world_size()
    with torch.no_grad():
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)] 
        dist.all_gather(gather_list, tensor)
    for rank in range(world_size):
        gather_list[rank] = tensor
    return torch.cat(gather_list, dim=0) 

def clip_forward(model, batch, device):
    images = batch['image'].to(device)
    texts = batch['text'].to(device)        

    image_features = model.module.encode_image(images)
    text_features = model.module.encode_text(texts)
    
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    all_image_features = gather_tensor(image_features)
    all_text_features = gather_tensor(text_features)
    
    logit_scale = model.module.logit_scale.exp()
    logits = logit_scale * all_image_features @ all_text_features.t()
    labels = torch.arange(logits.shape[0]).long().to(logits.device)
    
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.t(), labels)
    loss = (loss_i + loss_t) / 2

    return loss

def clip_forward_osm(model, batch, device):
    model, osm_model = model.module.model, model.module.osm_model

    images = batch['image'].to(device)
    texts = batch['text'].to(device)
    osm = batch['osm_text'].to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
    osm_features = osm_model.encode_text(osm)
    
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    osm_features = osm_features / osm_features.norm(dim=-1, keepdim=True)
    all_image_features = gather_tensor(image_features)
    all_text_features = gather_tensor(text_features)
    all_osm_features = gather_tensor(osm_features)
    
    logit_scale = model.logit_scale.exp()
    logit_scale_osm = osm_model.logit_scale.exp()
    logits = logit_scale * all_image_features @ all_text_features.t()
    logits_osm = logit_scale_osm * all_osm_features @ (all_text_features.detach().t())
    labels = torch.arange(logits.shape[0]).long().to(logits.device)
    
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.t(), labels)
    loss = (loss_i + loss_t) / 2

    loss_osm_i = nn.CrossEntropyLoss()(logits_osm, labels)
    loss_osm_t = nn.CrossEntropyLoss()(logits_osm.t(), labels)
    loss_osm = (loss_osm_i + loss_osm_t) / 2
    loss = loss_osm

    return loss

def siglip_forward(model, batch, device):
    images = batch['image'].to(device)
    texts = batch['text'].to(device)
    output = model.module(texts, images, return_loss=True)
    loss = output["loss"]
    
    return loss

def blip_forward(model, batch, device):
    images = batch['image'].to(device)
    texts = batch['text'].to(device)
    pad_token_id = 0
    attention_mask = (texts != pad_token_id).long()
    output = model.module.forward_(texts, images, return_loss=True, attention_mask=attention_mask)
    texts = texts
    loss = output["loss"]

    return loss
