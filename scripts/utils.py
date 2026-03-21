import torch
import torch.nn as nn
import random
import os
from torch.optim import AdamW
from torch.utils.data import Dataset
import pandas as pd
import timm
import numpy as np
from transformers import AutoModel, AutoTokenizer
from scripts.dataset import MultimodalDataset, get_transforms, collate_fn
from functools import partial
import torchmetrics
from torch.utils.data import DataLoader

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        
        self.regression = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 1),
            nn.ReLU()
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused_emb = image_emb * text_emb * mass.unsqueeze(1)

        logits = self.regression(fused_emb)
        return logits
    
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False

def validate(model, val_loader, device, mae_metric, mse_metric):
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label'].to(device)

            logits = model(**inputs)
            logits = logits.flatten()
            _ = mae_metric(preds=logits, target=labels)
            _ = mse_metric(preds=logits, target=labels)

    return {'MAE':mae_metric.compute().cpu().numpy(), 
            'MSE':mse_metric.compute().cpu().numpy()}

def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.regression.parameters(),
        'lr': config.REGRES_LR
    }])

    criterion = nn.L1Loss()#MSELoss

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")

    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="val")

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    # инициализируем метрику
    mse_metric_train = torchmetrics.MeanSquaredError(num_outputs=1).to(device)
    mse_metric_val = torchmetrics.MeanSquaredError(num_outputs=1).to(device)

    mae_metric_train = torchmetrics.MeanAbsoluteError(num_outputs=1).to(device)
    mae_metric_val = torchmetrics.MeanAbsoluteError(num_outputs=1).to(device)

    best_mae_val = 1e10

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label'].to(device)

            # Forward
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits.flatten(), labels)

            # Backward
            loss.backward()
            optimizer.step()
            logits = logits.flatten()
            _ = mse_metric_train(preds=logits, target=labels)
            _ = mae_metric_train(preds=logits, target=labels)
        
        train_mse = mse_metric_train.compute().cpu().numpy()
        train_mae = mae_metric_train.compute().cpu().numpy()
        mse_metric_train.reset()
        mae_metric_train.reset()
        # Валидация
        
        val_metrics = validate(model, val_loader, device, mae_metric_val, mse_metric_val)
        mse_metric_val.reset()
        mae_metric_val.reset()
        

        print(f"Epoch {epoch}/{config.EPOCHS-1} | MAE train: {train_mae:.2f} | MSE train: {train_mse :.2f}")
        print(f"MAE val: {val_metrics['MAE']:.2f} | MSE val: {val_metrics['MSE'] :.2f}")


        if val_metrics['MAE'].item() < best_mae_val:
            print(f"New best model, epoch: {epoch}")
            best_mae_val = val_metrics['MAE']
            torch.save(model.state_dict(), config.SAVE_PATH)

def inference(config, device, type_ds='val'):
    # грузим модель
    model = MultimodalModel(config)
    model.load_state_dict(torch.load(config.SAVE_PATH, weights_only=True)) 
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # подготавливаем данные для инференса
    test_transforms = get_transforms(config, ds_type=type_ds)
    val_dataset = MultimodalDataset(config, test_transforms, ds_type=type_ds)
    test_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))
    model.eval()
    df_res = pd.DataFrame()
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label']
            dish_id = batch['dish_id']

            logits = model(**inputs)
            logits = logits.flatten().to('cpu')
            df_res = pd.concat([df_res, pd.DataFrame({'target':labels, 'result':logits, 'dish_id':dish_id})])

    return df_res