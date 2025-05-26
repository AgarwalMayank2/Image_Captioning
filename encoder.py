from torchvision import transforms
import torch.nn as nn
class extractImageFeature:
    def __init__(self, data):
        self.data = data
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        

class encode:
    def __init__(self):
