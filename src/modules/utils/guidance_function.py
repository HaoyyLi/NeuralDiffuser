import os
import sys
project_path = '/home/src/NeuralDiffuser'
sys.path.append(project_path)
sys.path.append(project_path+'/src')
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from modules.packages import model_options as MO
from modules.packages import feature_extraction_detach as FE

def retrieve_clip_model(model_name):
    import clip;
    model, _ = clip.load(model_name, device='cpu')
    return model.visual

transform_for_CLIP = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225))
])

def get_model():
    model_string = 'ViT-B/32_clip'
    model_option = MO.get_model_options()[model_string]
    image_transforms = MO.get_recommended_transforms(model_string)
    model_name = model_option['model_name']
    train_type = model_option['train_type']

    model = eval(model_option['call'])
    model = model.eval()
    return model

get_feature_maps = FE.get_feature_maps

clip_transform_reconstruction = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225))
])
