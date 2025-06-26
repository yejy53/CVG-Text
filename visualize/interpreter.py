import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
from clip.clip import _tokenizer

def normalize_weights(attention_weights):
    max_weight = max(attention_weights)
    if max_weight > 0:
        return [w / max_weight for w in attention_weights]
    return attention_weights
  
def display_attention_as_image(words, weights, filename='output.png'):
    if font_path:
        prop = font_manager.FontProperties(fname=font_path)
    else:
        prop = None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    cmap = plt.cm.get_cmap('YlOrRd')

    x, y = 0.05, 0.8
    max_width = 0.95
    char_width = 0.02
    assert(len(words)==len(weights))

    for word, weight in zip(words, weights):
        color = cmap(weight)
        text = ax.text(x, y, word + " ", fontproperties=prop, fontsize=14, va='center', color='black',
                       bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2'))
        word_width = len(word) * char_width
        x += word_width

        if x > max_width:
            x = 0.05
            y -= 0.12

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    
def interpret(images, texts, model, device, start_layer=-1, start_layer_text=-1):
    batch_size = texts.shape[0]
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
        start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    
    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1: 
        start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text
   
    return text_relevance, image_relevance


def show_image_relevance(image_relevance, image, orig_image, output_path):
    def show_cam_on_image(img, mask):
        img = cv2.resize(img,(224,224))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    
    fig, axs = plt.subplots(1, 1)

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')

    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())

    image = image.permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs.imshow(vis)
    axs.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def show_text_relevance(text, text_encoding, R_text, output_path=None):
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    text_tokens=_tokenizer.encode(text)
    text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
    
    text_scores = text_scores.cpu().numpy().tolist()
    text_scores = normalize_weights(text_scores)

    if output_path != None:
        display_attention_as_image(text_tokens_decoded, text_scores, output_path)
    return text_scores, text_tokens_decoded