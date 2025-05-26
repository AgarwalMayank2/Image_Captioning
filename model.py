import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN = False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        resnet = models.resnet50(pretrained = True)