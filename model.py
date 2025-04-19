import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self):
        """super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained = True) ## using inception here because it is best in pattern recognition
        self.inception_fc = nn.Linear(2048, embed_size) ## the output of inception is 2048, so we need to change it to embed_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)"""



    def forward(self, images):
        features = self.inception(images)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN

        return self.dropout(self.relu(features))
