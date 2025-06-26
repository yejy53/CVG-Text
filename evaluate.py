import os
import torch
from tqdm import tqdm
import numpy as np
import csv
from math import radians, sin, cos, sqrt, atan2
import re
import json
from torch.nn import functional as F
from transformers import ViltProcessor

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def extract_lat_lon(filename):
    base_name = os.path.basename(filename)
    match = re.match(r"([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)", base_name)
    lat, lng = float(match.group(1)), float(match.group(2))
    return lat, lng

def evaluate_clip(model, dataloader, device, output_csv_path=None, output_metrics_path=None):
    model.to(device)
    model.eval()

    image_features_list = []
    text_features_list = []
    image_filenames_list = []
    text_filenames_list = []
    texts_list = []

    print("Calculating image and text features...")
    for batch in tqdm(dataloader, desc="Processing images and texts"):
        images = batch['image'].to(device)
        texts = batch['text'].to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

        # Append features and filenames
        image_features_list.append(image_features.cpu())
        text_features_list.append(text_features.cpu())
        image_filenames_list.extend(batch['image_path'])
        text_filenames_list.extend(batch['image_path'])
        texts_list.extend(batch['original_text'])

    # Convert lists to tensors
    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
        
    normalized_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarities = normalized_image_features @ normalized_text_features.t()

    return cal_score(similarities, image_filenames_list, text_filenames_list,texts_list, output_csv_path, output_metrics_path)

def evaluate_clip_osm(model, dataloader, device, output_csv_path=None, output_metrics_path=None):
    model, osm_model = model.model, model.osm_model
    model.to(device)
    osm_model.to(device)
    model.eval()
    osm_model.eval()

    image_features_list = []
    text_features_list = []
    osm_features_list = []
    image_filenames_list = []
    text_filenames_list = []
    texts_list = []

    print("Calculating image and text features...")
    for batch in tqdm(dataloader, desc="Processing images and texts"):
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        osm = batch['osm_text'].to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            osm_features = osm_model.encode_text(osm)
        # Append features and filenames
        image_features_list.append(image_features.cpu())
        text_features_list.append(text_features.cpu())
        osm_features_list.append(osm_features.cpu())
        image_filenames_list.extend(batch['image_path'])
        text_filenames_list.extend(batch['image_path'])
        texts_list.extend(batch['original_text'])

    # Convert lists to tensors
    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    osm_features = torch.cat(osm_features_list, dim=0)
        
    normalized_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    normalized_osm_features = osm_features / osm_features.norm(dim=-1, keepdim=True)
    similarities = normalized_image_features @ normalized_text_features.t()
    sim_osm = normalized_osm_features @ normalized_text_features.t()
    similarities = sim_osm + similarities

    return cal_score(similarities, image_filenames_list, text_filenames_list,texts_list, output_csv_path, output_metrics_path)

def evaluate_siglip(model, dataloader, device, output_csv_path=None, output_metrics_path=None):
    model.to(device)
    model.eval()

    image_features_list = []
    text_features_list = []
    image_filenames_list = []
    text_filenames_list = []
    texts_list = []

    print("Calculating image and text features...")
    for batch in tqdm(dataloader, desc="Processing images and texts"):
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        #pad_token_id = 0
        #attention_mask = (texts != pad_token_id).long().squeeze(1)
        with torch.no_grad():
            image_features = model.get_image_features(images)
            text_features = model.get_text_features(texts)#, attention_mask=attention_mask)

        image_features_list.append(image_features)
        text_features_list.append(text_features)
        image_filenames_list.extend(batch['image_path'])
        text_filenames_list.extend(batch['image_path'])
        texts_list.extend(batch['original_text'])

    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    normalized_image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    normalized_text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    print("Calculating similarity matrix...")
    logits_per_text = torch.matmul(normalized_text_features, normalized_image_features.t()) * model.logit_scale.exp() + model.logit_bias
    logits_per_image = logits_per_text.t()
    similarities = torch.sigmoid(logits_per_image).detach().cpu()
    return cal_score(similarities, image_filenames_list, text_filenames_list,texts_list, output_csv_path, output_metrics_path)

def evaluate_blip(model, dataloader, device, output_csv_path=None, output_metrics_path=None, k = 15):
    model.to(device)
    model.eval()
    image_list = []
    text_id_list = []
    image_features_list = []
    text_features_list = []
    image_filenames_list = []
    text_filenames_list = []
    texts_list = []
    print("Calculating image and text features...")
    for batch in tqdm(dataloader, desc="Processing images and texts"):
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        pad_token_id = 0
        attention_mask = (texts != pad_token_id).long()

        with torch.no_grad():
            image_features = model.get_image_features(images)
            text_features = model.get_text_features(texts, attention_mask=attention_mask)
        image_list.append(images)
        text_id_list.append(texts)
        image_features_list.append(image_features)
        text_features_list.append(text_features)
        image_filenames_list.extend(batch['image_path'])
        text_filenames_list.extend(batch['image_path'])
        texts_list.extend(batch['original_text'])
    image_values = torch.cat(image_list, dim=0)
    text_ids = torch.cat(text_id_list, dim=0)
    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    normalized_image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    normalized_text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    similarities = normalized_image_features @ normalized_text_features.t()

    if k == 0 :
        return cal_score(similarities, image_filenames_list, text_filenames_list,texts_list, output_csv_path, output_metrics_path)
    else: 
        print(f"Performing ITM on the top-{k} results for each text...")

        for i, text in enumerate(tqdm(text_ids, desc="ITM Stage")):
            top_k_indices = similarities[i].topk(k=k, dim=-1).indices
            top_k_images = image_values[top_k_indices]
            texts = text.repeat(k,1)
            with torch.no_grad():
                pad_token_id = 0
                attention_mask = (texts != pad_token_id).long()
                itm_output = model(texts, top_k_images, use_itm_head=True, attention_mask=attention_mask)['itm_score']
                itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]

                similarities[top_k_indices, i] += itm_score

        return cal_score(similarities, image_filenames_list, text_filenames_list,texts_list, output_csv_path, output_metrics_path)

def evaluate_albef(model, dataloader, device, output_csv_path=None, k = 15, output_metrics_path=None):
    model.to(device)
    model.eval()
    image_list = []
    text_id_list = []
    image_features_list = []
    text_features_list = []
    image_filenames_list = []
    text_filenames_list = []
    texts_list = []
    print("Calculating image and text features...")
    
    for batch in tqdm(dataloader, desc="Processing images and texts"):
        images = batch['image'].to(device)
        texts = batch['text'].to(device)

        text_output = model.text_encoder(texts, attention_mask = texts.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        
        image_feat = model.visual_encoder(images)        
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)

        image_list.append(images)
        text_id_list.append(texts)

        image_features_list.append(image_features)
        text_features_list.append(text_embed)
        image_filenames_list.extend(batch['image_path'])
        text_filenames_list.extend(batch['image_path'])
        texts_list.extend(batch['original_text'])

    image_values = torch.cat(image_list, dim=0)
    text_ids = torch.cat(text_id_list, dim=0)
    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)

    similarities = image_features @ text_features.t()

    if k == 0 :
        return cal_score(similarities, image_filenames_list, text_filenames_list,texts_list, output_csv_path, output_metrics_path)
    else: 
        print(f"Performing ITM on the top-{k} results for each text...")

        for i, text in enumerate(tqdm(text_ids, desc="ITM Stage")):
            top_k_indices = similarities[i].topk(k=k, dim=-1).indices
            top_k_images = image_values[top_k_indices]
            texts = text.repeat(k,1)
            with torch.no_grad():
                pad_token_id = 0
                attention_mask = (texts != pad_token_id).long()
                encoder_att = torch.ones(top_k_images.size()[:-1],dtype=torch.long).to(device)
                output = model.text_encoder(encoder_embeds = texts, 
                                    attention_mask = attention_mask,
                                    encoder_hidden_states = top_k_images,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
                itm_output = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
                #itm_output = model(texts, top_k_images, use_itm_head=True, attention_mask=attention_mask)['itm_score']
                itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]

                similarities[top_k_indices, i] += itm_score

        return cal_score(similarities, image_filenames_list, text_filenames_list,texts_list, output_csv_path, output_metrics_path)

def evaluate_vilt_(model, dataloader, device, output_csv_path=None, output_metrics_path=None):
    model.to(device)
    model.eval()

    def compute_recall_at_k(gt_idx, retrieved_indices, k):
        return int(gt_idx in retrieved_indices[:k])

    from PIL import Image
    offset, recall_50, recall_100, recall_150 = 0, 0, 0, 0
    recall_at_1, recall_at_5, recall_at_10 = 0, 0, 0
    total = 0
    processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-coco')
    k = 100
    images = dataloader.dataset.image  # [:k]
    texts = dataloader.dataset.text  # [:k]

    for i, text in enumerate(tqdm(texts)):
        original_lat, original_lon = extract_lat_lon(images[i])
        distances = []
        for j, image_filename in enumerate(images):
            lat, lon = extract_lat_lon(image_filename)
            distance = haversine(original_lat, original_lon, lat, lon)
            distances.append((distance, j))
        distances.sort(key=lambda x: x[0])
        closest_indices = [idx for _, idx in distances[:100]]

        closest_images = [Image.open(images[i]).convert("RGB") for i in closest_indices]
        encoding = processor(closest_images, [text] * 100, max_length=40, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoding)
        scores = outputs.logits.squeeze(1)
        sorted_indices = torch.argsort(scores, descending=True).tolist()

        top1_idx = closest_indices[sorted_indices[0]]
        top1_image = images[top1_idx]
        top1_lat, top1_lon = extract_lat_lon(top1_image)
        top1_distance = haversine(original_lat, original_lon, top1_lat, top1_lon)
        offset += top1_distance

        if top1_distance <= 0.05:
            recall_50 += 1

        if top1_distance <= 0.1:
            recall_100 += 1

        if top1_distance <= 0.15:
            recall_150 += 1

        recall_at_1 += compute_recall_at_k(0, sorted_indices, 1)
        recall_at_5 += compute_recall_at_k(0, sorted_indices, 5)
        recall_at_10 += compute_recall_at_k(0, sorted_indices, 10)
        total += 1

    overall_r_at_1 = (recall_at_1 / total) * 100 if total > 0 else 0
    overall_r_at_5 = (recall_at_5 / total) * 100 if total > 0 else 0
    overall_r_at_10 = (recall_at_10 / total) * 100 if total > 0 else 0
    distance_top1 = (offset / total) if total > 0 else 0
    distance_recall_50 = (recall_50 / total) * 100 if total > 0 else 0
    distance_recall_100 = (recall_100 / total) * 100 if total > 0 else 0
    distance_recall_150 = (recall_150 / total) * 100 if total > 0 else 0

    print(f"R@1: {overall_r_at_1 :.4f}, R@5: {overall_r_at_5:.4f}, R@10: {overall_r_at_10:.4f}")
    print(f"Distance Top-1: {distance_top1:.4f}km")
    print(f"L@50 (within 50m): {distance_recall_50:.4f}%")
    print(f"L@100 (within 100m): {distance_recall_100:.4f}%")
    print(f"L@150 (within 150m): {distance_recall_150:.4f}%")

    if output_metrics_path is not None:
        with open(output_metrics_path, 'w') as metrics_file:
            metrics_file.write(f"Overall R@1: {overall_r_at_1:.2f}%\n")
            metrics_file.write(f"Overall R@5: {overall_r_at_5:.2f}%\n")
            metrics_file.write(f"Overall R@10: {overall_r_at_10:.2f}%\n")
            metrics_file.write(f"Average Geographic Deviation: {distance_top1:.2f} km\n")
            metrics_file.write(f"L@50 (within 50m): {distance_recall_50:.2f} %\n")
            metrics_file.write(f"L@100 (within 100m): {distance_recall_100:.2f} %\n")
            metrics_file.write(f"L@150 (within 150m): {distance_recall_150:.2f} %\n")
        print(f"Metrics saved to {output_metrics_path}")

    return overall_r_at_1, overall_r_at_5, overall_r_at_10

def evaluate_vilt(model, dataloader, device, output_csv_path=None, output_metrics_path=None):
    model.to(device)
    model.eval()

    def compute_recall_at_k(gt_idx, retrieved_indices, k):
        return int(gt_idx in retrieved_indices[:k])

    def is_panorama(filename):
        """
        Determine if the file is a panoramic image.
        Panoramic filenames generally have more metadata segments compared to single-view images.
        """
        parts = filename.split("_")
        return len(parts) > 2

    from PIL import Image
    processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-coco')

    # Initialize metrics
    metrics = {
        "overall": {"R@1": 0, "R@5": 0, "R@10": 0, "L@50": 0, "L@100": 0, "L@150": 0, "Deviation": 0, "Count": 0},
        "panorama": {"R@1": 0, "R@5": 0, "R@10": 0, "L@50": 0, "L@100": 0, "L@150": 0, "Deviation": 0, "Count": 0},
        "single_view": {"R@1": 0, "R@5": 0, "R@10": 0, "L@50": 0, "L@100": 0, "L@150": 0, "Deviation": 0, "Count": 0},
    }

    images = dataloader.dataset.image
    texts = dataloader.dataset.text

    for i, text in enumerate(tqdm(texts)):
        original_lat, original_lon = extract_lat_lon(images[i])
        distances = []
        for j, image_filename in enumerate(images):
            lat, lon = extract_lat_lon(image_filename)
            distance = haversine(original_lat, original_lon, lat, lon)
            distances.append((distance, j))
        distances.sort(key=lambda x: x[0])
        closest_indices = [idx for _, idx in distances[:100]]

        closest_images = [Image.open(images[i]).convert("RGB") for i in closest_indices]
        encoding = processor(closest_images, [text] * 100, max_length=40, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoding)
        scores = outputs.logits.squeeze(1)
        sorted_indices = torch.argsort(scores, descending=True).tolist()

        top1_idx = closest_indices[sorted_indices[0]]
        top1_image = images[top1_idx]
        top1_lat, top1_lon = extract_lat_lon(top1_image)
        top1_distance = haversine(original_lat, original_lon, top1_lat, top1_lon)

        # Determine if it's a panoramic image
        is_pano = is_panorama(images[i])
        metric_key = "panorama" if is_pano else "single_view"

        # Update metrics for both specific and overall groups
        for key in ["overall", metric_key]:
            metrics[key]["Count"] += 1
            metrics[key]["Deviation"] += top1_distance
            metrics[key]["L@50"] += int(top1_distance <= 0.05)
            metrics[key]["L@100"] += int(top1_distance <= 0.1)
            metrics[key]["L@150"] += int(top1_distance <= 0.15)
            metrics[key]["R@1"] += compute_recall_at_k(0, sorted_indices, 1)
            metrics[key]["R@5"] += compute_recall_at_k(0, sorted_indices, 5)
            metrics[key]["R@10"] += compute_recall_at_k(0, sorted_indices, 10)

    def calculate_percentages(metrics):
        count = metrics["Count"]
        if count == 0:
            return metrics
        return {
            "R@1": (metrics["R@1"] / count) * 100,
            "R@5": (metrics["R@5"] / count) * 100,
            "R@10": (metrics["R@10"] / count) * 100,
            "L@50": (metrics["L@50"] / count) * 100,
            "L@100": (metrics["L@100"] / count) * 100,
            "L@150": (metrics["L@150"] / count) * 100,
            "Deviation": metrics["Deviation"] / count,
            "Count": count,
        }

    metrics["overall"] = calculate_percentages(metrics["overall"])
    metrics["panorama"] = calculate_percentages(metrics["panorama"])
    metrics["single_view"] = calculate_percentages(metrics["single_view"])

    # Print results
    print("\nOverall Results:")
    print(f"R@1: {metrics['overall']['R@1']:.2f}%")
    print(f"R@5: {metrics['overall']['R@5']:.2f}%")
    print(f"R@10: {metrics['overall']['R@10']:.2f}%")
    print(f"Avg Deviation: {metrics['overall']['Deviation']:.2f} km")
    print(f"L@50: {metrics['overall']['L@50']:.2f}%")
    print(f"L@100: {metrics['overall']['L@100']:.2f}%")
    print(f"L@150: {metrics['overall']['L@150']:.2f}%")

    print("\nPanorama Results:")
    print(f"R@1: {metrics['panorama']['R@1']:.2f}%")
    print(f"R@5: {metrics['panorama']['R@5']:.2f}%")
    print(f"R@10: {metrics['panorama']['R@10']:.2f}%")
    print(f"Avg Deviation: {metrics['panorama']['Deviation']:.2f} km")
    print(f"L@50: {metrics['panorama']['L@50']:.2f}%")
    print(f"L@100: {metrics['panorama']['L@100']:.2f}%")
    print(f"L@150: {metrics['panorama']['L@150']:.2f}%")
    print(f"Num: {metrics['panorama']['Count']}")

    print("\nSingle-View Results:")
    print(f"R@1: {metrics['single_view']['R@1']:.2f}%")
    print(f"R@5: {metrics['single_view']['R@5']:.2f}%")
    print(f"R@10: {metrics['single_view']['R@10']:.2f}%")
    print(f"Avg Deviation: {metrics['single_view']['Deviation']:.2f} km")
    print(f"L@50: {metrics['single_view']['L@50']:.2f}%")
    print(f"L@100: {metrics['single_view']['L@100']:.2f}%")
    print(f"L@150: {metrics['single_view']['L@150']:.2f}%")
    print(f"Num: {metrics['single_view']['Count']}")

    # Save metrics to file if needed
    if output_metrics_path is not None:
        with open(output_metrics_path, 'w') as metrics_file:
            metrics_file.write("Overall Results:\n")
            for key, value in metrics["overall"].items():
                metrics_file.write(f"{key}: {value:.2f}\n" if isinstance(value, float) else f"{key}: {value}\n")
            metrics_file.write("\nPanorama Results:\n")
            for key, value in metrics["panorama"].items():
                metrics_file.write(f"{key}: {value:.2f}\n" if isinstance(value, float) else f"{key}: {value}\n")
            metrics_file.write("\nSingle-View Results:\n")
            for key, value in metrics["single_view"].items():
                metrics_file.write(f"{key}: {value:.2f}\n" if isinstance(value, float) else f"{key}: {value}\n")
        print(f"Metrics saved to {output_metrics_path}")

    return metrics["overall"]["R@1"], metrics["overall"]["R@5"], metrics["overall"]["R@10"]

def cal_score_(similarities, image_filenames_list, text_filenames_list,texts_list, output_csv_path=None,output_metrics_path=None, m=100):
    top_k_records = []
    total_deviation = 0
    total_recall_50 = 0
    total_recall_100 = 0
    print("Calculating R@1, R@5, R@10...")
    total_samples = len(text_filenames_list)
    r_at_1 = 0
    r_at_5 = 0
    r_at_10 = 0

    for i in range(total_samples):
        original_lat, original_lon = extract_lat_lon(text_filenames_list[i])

        distances = []
        for j, image_filename in enumerate(image_filenames_list):
            lat, lon = extract_lat_lon(image_filename)
            distance = haversine(original_lat, original_lon, lat, lon)
            distances.append((distance, j))

        distances.sort(key=lambda x: x[0])
        closest_indices = [idx for _, idx in distances[:m]]
        sim = similarities[closest_indices, i].cpu().numpy()
        sorted_indices = np.argsort(-sim)
        best_matches = [os.path.basename(image_filenames_list[closest_indices[idx]]) for idx in sorted_indices]
        original_image_name = os.path.basename(text_filenames_list[i])

        best_matches = list(dict.fromkeys(best_matches))
        label = best_matches[0] == original_image_name

        deviation = 0
        if best_matches[0] == original_image_name:
            deviation = 0  # No deviation if the first match is the correct one
        else:
            # Calculate the distance between the original image and the first match
            lat_match, lon_match = extract_lat_lon(image_filenames_list[closest_indices[sorted_indices[0]]])
            deviation = haversine(original_lat, original_lon, lat_match, lon_match)

        if deviation <= 0.05:
            total_recall_50 += 1
        if deviation <= 0.1:
            total_recall_100 += 1
        total_deviation += deviation
        top_k_records.append({
            'original_image': original_image_name,
            'text': texts_list[i],
            'top_5_matches': best_matches[:5],
            'label': label
        })

        if original_image_name == best_matches[0]:
            r_at_1 += 1

        if original_image_name in best_matches[:5]:
            r_at_5 += 1

        if original_image_name in best_matches[:10]:
            r_at_10 += 1

    # Overall recall statistics
    overall_r_at_1 = (r_at_1 / total_samples) * 100 if total_samples > 0 else 0
    overall_r_at_5 = (r_at_5 / total_samples) * 100 if total_samples > 0 else 0
    overall_r_at_10 = (r_at_10 / total_samples) * 100 if total_samples > 0 else 0
    avg_deviation = total_deviation / total_samples if total_samples > 0 else 0
    avg_recall_50 = total_recall_50 / total_samples * 100 if total_samples > 0 else 0
    avg_recall_100 = total_recall_100 / total_samples * 100 if total_samples > 0 else 0
    print("\nOverall Results:")
    print(f"Overall R@1: {overall_r_at_1:.2f}%")
    print(f"Overall R@5: {overall_r_at_5:.2f}%")
    print(f"Overall R@10: {overall_r_at_10:.2f}%")
    print(f"Average Geographic Deviation: {avg_deviation:.2f} km")
    print(f"Average Geographic Deviation R@1 50m: {avg_recall_50:.2f} %")
    print(f"Average Geographic Deviation R@1 100m: {avg_recall_100:.2f} %")
    if output_metrics_path is not None:
        with open(output_metrics_path, 'w') as metrics_file:
            metrics_file.write(f"Overall R@1: {overall_r_at_1:.2f}%\n")
            metrics_file.write(f"Overall R@5: {overall_r_at_5:.2f}%\n")
            metrics_file.write(f"Overall R@10: {overall_r_at_10:.2f}%\n")
            metrics_file.write(f"Average Geographic Deviation: {avg_deviation:.2f} km\n")
            metrics_file.write(f"Average Geographic Deviation R@1 50m: {avg_recall_50:.2f} %\n")
            metrics_file.write(f"Average Geographic Deviation R@1 100m: {avg_recall_100:.2f} %\n")
        print(f"Metrics saved to {output_metrics_path}")

    if output_csv_path is not None:
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['original_image', 'text', 'top_5_matches','label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for record in top_k_records:
                writer.writerow({
                    'original_image': record['original_image'],
                    'text': record['text'],
                    'top_5_matches': json.dumps(record['top_5_matches']),
                    'label': record['label']
                })

        print(f"Top 5 matching information saved to {output_csv_path}")
    return overall_r_at_1, overall_r_at_5, overall_r_at_10

def cal_score(similarities, image_filenames_list, text_filenames_list, texts_list, output_csv_path=None, output_metrics_path=None, m=100):
    def is_panorama(filename):
        """
        Determine if the file is a panoramic image.
        Panoramic filenames generally have more metadata segments compared to single-view images.
        """
        parts = filename.split("_")
        return len(parts) > 2

    # Metrics for all, panoramas, and single views
    overall_metrics = {"R@1": 0, "R@5": 0, "R@10": 0, "L@50": 0, "L@100": 0, "L@150": 0, "Deviation": 0, "Count": 0}
    panorama_metrics = {"R@1": 0, "R@5": 0, "R@10": 0, "L@50": 0, "L@100": 0, "L@150": 0, "Deviation": 0, "Count": 0}
    single_view_metrics = {"R@1": 0, "R@5": 0, "R@10": 0, "L@50": 0, "L@100": 0, "L@150": 0, "Deviation": 0, "Count": 0}

    total_samples = len(text_filenames_list)
    top_k_records = []

    print("Calculating R@1, R@5, R@10...")

    for i in range(total_samples):
        original_lat, original_lon = extract_lat_lon(text_filenames_list[i])
        is_pano = is_panorama(text_filenames_list[i])

        distances = []
        for j, image_filename in enumerate(image_filenames_list):
            lat, lon = extract_lat_lon(image_filename)
            distance = haversine(original_lat, original_lon, lat, lon)
            distances.append((distance, j))

        distances.sort(key=lambda x: x[0])
        closest_indices = [idx for _, idx in distances[:m]]
        sim = similarities[closest_indices, i].cpu().numpy()
        sorted_indices = np.argsort(-sim)
        best_matches = [os.path.basename(image_filenames_list[closest_indices[idx]]) for idx in sorted_indices]
        original_image_name = os.path.basename(text_filenames_list[i])

        best_matches = list(dict.fromkeys(best_matches))
        label = best_matches[0] == original_image_name

        deviation = 0
        if best_matches[0] == original_image_name:
            deviation = 0
        else:
            lat_match, lon_match = extract_lat_lon(image_filenames_list[closest_indices[sorted_indices[0]]])
            deviation = haversine(original_lat, original_lon, lat_match, lon_match)

        # Update recall statistics
        l_50 = deviation <= 0.05
        l_100 = deviation <= 0.1
        l_150 = deviation <= 0.15

        for metrics in [overall_metrics, panorama_metrics if is_pano else single_view_metrics]:
            metrics["Count"] += 1
            metrics["Deviation"] += deviation
            metrics["L@50"] += int(l_50)
            metrics["L@100"] += int(l_100)
            metrics["L@150"] += int(l_150)
            metrics["R@1"] += int(best_matches[0] == original_image_name)
            metrics["R@5"] += int(original_image_name in best_matches[:5])
            metrics["R@10"] += int(original_image_name in best_matches[:10])

        # Record data for CSV output
        top_k_records.append({
            'original_image': original_image_name,
            'text': texts_list[i],
            'top_5_matches': best_matches[:5],
            'label': label
        })

    # Aggregate statistics
    def calculate_average(metrics):
        count = metrics["Count"]
        if count == 0:
            return metrics
        return {
            "R@1": (metrics["R@1"] / count) * 100,
            "R@5": (metrics["R@5"] / count) * 100,
            "R@10": (metrics["R@10"] / count) * 100,
            "L@50": (metrics["L@50"] / count) * 100,
            "L@100": (metrics["L@100"] / count) * 100,
            "L@150": (metrics["L@150"] / count) * 100,
            "Deviation": metrics["Deviation"] / count,
            "Count": count
        }

    overall_metrics = calculate_average(overall_metrics)
    panorama_metrics = calculate_average(panorama_metrics)
    single_view_metrics = calculate_average(single_view_metrics)

    print("\nOverall Results:")
    print(f"R@1: {overall_metrics['R@1']:.2f}%")
    print(f"R@5: {overall_metrics['R@5']:.2f}%")
    print(f"R@10: {overall_metrics['R@10']:.2f}%")
    print(f"Avg Deviation: {overall_metrics['Deviation']:.2f} km")
    print(f"L@50: {overall_metrics['L@50']:.2f}%")
    print(f"L@100: {overall_metrics['L@100']:.2f}%")
    print(f"L@150: {overall_metrics['L@150']:.2f}%")
    
    print("\nPanorama Results:")
    print(f"R@1: {panorama_metrics['R@1']:.2f}%")
    print(f"R@5: {panorama_metrics['R@5']:.2f}%")
    print(f"R@10: {panorama_metrics['R@10']:.2f}%")
    print(f"Avg Deviation: {panorama_metrics['Deviation']:.2f} km")
    print(f"L@50: {panorama_metrics['L@50']:.2f}%")
    print(f"L@100: {panorama_metrics['L@100']:.2f}%")
    print(f"L@150: {panorama_metrics['L@150']:.2f}%")
    print(f"Num: {panorama_metrics['Count']}")

    print("\nSingle-View Results:")
    print(f"R@1: {single_view_metrics['R@1']:.2f}%")
    print(f"R@5: {single_view_metrics['R@5']:.2f}%")
    print(f"R@10: {single_view_metrics['R@10']:.2f}%")
    print(f"Avg Deviation: {single_view_metrics['Deviation']:.2f} km")
    print(f"L@50: {single_view_metrics['L@50']:.2f}%")
    print(f"L@100: {single_view_metrics['L@100']:.2f}%")
    print(f"L@150: {single_view_metrics['L@150']:.2f}%")
    print(f"Num: {single_view_metrics['Count']}")

    # Save results to files if needed
    if output_metrics_path:
        with open(output_metrics_path, 'w') as metrics_file:
            metrics_file.write("Overall Results:\n")
            for key, value in overall_metrics.items():
                metrics_file.write(f"{key}: {value:.2f}\n" if isinstance(value, float) else f"{key}: {value}\n")
            metrics_file.write("\nPanorama Results:\n")
            for key, value in panorama_metrics.items():
                metrics_file.write(f"{key}: {value:.2f}\n" if isinstance(value, float) else f"{key}: {value}\n")
            metrics_file.write("\nSingle-View Results:\n")
            for key, value in single_view_metrics.items():
                metrics_file.write(f"{key}: {value:.2f}\n" if isinstance(value, float) else f"{key}: {value}\n")
        print(f"Metrics saved to {output_metrics_path}")

    if output_csv_path:
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['original_image', 'text', 'top_5_matches', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in top_k_records:
                writer.writerow({
                    'original_image': record['original_image'],
                    'text': record['text'],
                    'top_5_matches': json.dumps(record['top_5_matches']),
                    'label': record['label']
                })
        print(f"Top 5 matching information saved to {output_csv_path}")

    return overall_metrics['R@1'], overall_metrics['R@5'], overall_metrics['R@10']


