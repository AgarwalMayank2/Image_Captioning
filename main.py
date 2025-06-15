import pre_processing
from sklearn.model_selection import train_test_split
import torch
import resnet_encoder
import pandas as pd
import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

processed_data = pre_processing.PreProcess('flickr8k/captions.txt')

unq_data = processed_data.data[['image']].drop_duplicates()

#resnet_encoder.get_feature(unq_data, 'flickr8k/Images', device, batch_size=1)

train.trainer(processed_data, device)