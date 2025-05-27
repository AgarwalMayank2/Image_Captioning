from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.models import resnet50
from torch.autograd import Variable
import torch
import pickle

class extractImageFeature:
    def __init__(self, data):
        self.data = data
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['image']
        image_loc = 'flickr8k/Images/'+image_name

        img = Image.open(image_loc)
        transformed_img = self.transforms(img)

        return image_name, transformed_img


def get_dataloader(data, batch_size = 1):
    image_dataset = extractImageFeature(data)
    image_dataloader = DataLoader(image_dataset, batch_size = batch_size, shuffle = False)
    return image_dataloader


class encode:
    def __init__(self, device):
        self.device = device
        self.resnet = resnet50(pretrained = True).to(device)
        self.resnet_layer4 = self.resnet._modules.get('layer4').to(device)

    def get_vector(self, image):
        image = Variable(image)
        my_embedding = torch.zeros(1, 512, 7, 7)
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        h = self.resnet_layer4.register_forward_hook(copy_data)
        self.resnet(image)
        return my_embedding
    

def get_feature(data, device, batch_size = 1):
    dataloader = get_dataloader(data, batch_size)

    image_feature = {}
    for image_name, image in dataloader:
        image = image.to(device)
        embedding = encode(device).get_vector(image)

        image_feature[image_name[0]] = embedding

    encoder_file = open("./EncodedImageResnet.pkl", "wb")
    pickle.dump(image_feature, encoder_file)
    encoder_file.close()