import os
import json
from PIL import Image
from torch.utils.data import Dataset
import random

class OSMTextImageDataset(Dataset):
    def __init__(self, image_root_dirs, json_file_paths, preprocessor=None, processor=None):
        self.image_root_dirs = image_root_dirs if isinstance(image_root_dirs, list) else [image_root_dirs]
        self.json_file_paths = json_file_paths if isinstance(json_file_paths, list) else [json_file_paths]
        self.preprocessor = preprocessor
        self.processor = processor

        self.image = []
        self.text = []
        self.img2txt = {}
        self.txt2img = {}

        for json_file_path, image_root_dir in zip(self.json_file_paths, self.image_root_dirs):
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            """
            for idx, item in enumerate(json_data):
                image_name = os.path.basename(item['bev_image'])
                caption = item['caption']

                self.image.append(os.path.join(image_root_dir, image_name))
                self.text.append(caption)
                
                self.img2txt[len(self.image) - 1] = [len(self.text) - 1]
                self.txt2img[len(self.text) - 1] = len(self.image) - 1
            """
            for idx, (image_name, caption) in enumerate(json_data.items()):

                self.image.append(os.path.join(image_root_dir, image_name))
                self.text.append(caption)

                self.img2txt[len(self.image) - 1] = [len(self.text) - 1]
                self.txt2img[len(self.text) - 1] = len(self.image) - 1

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        image_ = Image.open(self.image[idx]).convert("RGB")
        text = self.text[idx]

        if self.preprocessor:
            image, text_id = self.preprocessor(image_, text)

        if self.processor:
            output = self.processor(image_, text, return_tensors="pt", max_length=40, padding='max_length', truncation=True)
            for key, value in output.items():
                value = value.squeeze(0)
            return output
        
        
        return {
            'image': image,
            'text': text_id,
            'original_text': text,
            'image_path': self.image[idx],
            'idx': idx
        }
