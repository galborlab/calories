import torch
from torch.utils.data import Dataset
from PIL import Image
import timm
import numpy as np
import pandas as pd
import albumentations as A

class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train"):
        self.df_general = pd.read_csv(config.DF_GENERAL_PATH)

        self.df_product = pd.read_csv(config.DF_PRODUCT_PATH)
        self.df_product.id = 'ingr_' + self.df_product.id.astype(str).str.zfill(10)

        if ds_type == "train":
            self.df_general = self.df_general[self.df_general['split']=='train'].drop('split', axis=1).reset_index(drop=True)
        else:
            self.df_general = self.df_general[self.df_general['split']=='test'].drop('split', axis=1).reset_index(drop=True)

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.transforms = transforms
        self.image_path = config.IMAGE_PATH

    def __len__(self):
        return len(self.df_general)

    def __getitem__(self, idx):
        total_mass = self.df_general.loc[idx, "total_mass"]
        label = self.df_general.loc[idx, "total_calories"]

        ingredients = self.df_general.loc[idx, "ingredients"].split(';')
        ingredients = np.unique(ingredients)
        text = self.df_product.loc[self.df_product.id.isin(ingredients), 'ingr'].to_list()
        text = ', '.join(text)
        
        img_path = self.df_general.loc[idx, "dish_id"]

        try:
            image = Image.open(self.image_path + f"{img_path}/rgb.png").convert('RGB')
        except:
            image = torch.randint(0, 255, (*self.image_cfg.input_size[1:],
                                           self.image_cfg.input_size[0])).to(
                                               torch.float32)

        image = self.transforms(image=np.array(image))["image"]
        return {"label": label, "image": image, "text": text, 'mass':total_mass, 'dish_id':img_path}

def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.RandomCrop(height=cfg.input_size[1], 
                             width=cfg.input_size[2], p=1.0),
                A.SquareSymmetry(p=1.0),
                A.Affine(scale=(0.8, 1.2),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
                
                A.ColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              hue=0.1,
                              p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=config.SEED,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=config.SEED,
        )

    return transforms

def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    mass = torch.tensor([item["mass"] for item in batch], dtype=torch.float32)
    dish_id = [item["dish_id"] for item in batch]

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)

    return {
        "label": labels,
        "image": images,
        "mass": mass,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        'dish_id':dish_id
    }