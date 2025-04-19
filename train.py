import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from loader import flickrdataset, data_loader

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5,  0.5, 0.5))
        ]
    )

    freq_threshold = 2
    batch_size = 32

    dataset = flickrdataset(
        root_dir = "flickr8k/Images",
        captions_file = "flickr8k/captions.txt",
        transform = transform,
        freq_threshold = freq_threshold
    )

    train_loader = data_loader(
        root_dir = "flickr8k/Images",
        captions_file = "flickr8k/captions.txt",
        transform = transform,
        num_workers = 2,
        freq_threshold = freq_threshold,
        batch_size = batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 30