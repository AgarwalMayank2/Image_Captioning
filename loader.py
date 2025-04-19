import pandas as pd
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import cv2
import torch
from PIL import Image

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary: ## Class for tokenizing the text and creating a vocabulary
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        
        self.stoi = {v:k for k, v in self.itos.items()}

        self.freq_threshold = freq_threshold

    def __len__(self): return len(self.itos)

    def tokenize(self, text): return [token.text.lower() for token in spacy_eng.tokenizer(text)] ## using spacy for better tokenization

    def build_vocab(self, sentence_list):
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else: frequencies[word] +=1
        
        frequencies_desc = dict(sorted(frequencies.items(), key = lambda item: item[1], reverse = True))

        for k, v in frequencies_desc.items():
            if v<self.freq_threshold: continue
            self.stoi[k] = idx
            self.itos[idx] = k
            idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token is self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class flickrdataset(Dataset):
    def __init__(self, root_dir, captions_file, transform = None, freq_threshold = 2):
        self.root_dir = root_dir
        self.caption_file_data = pd.read_csv(captions_file)
        self.transform = transform

        self.imgs = self.caption_file_data["image"]
        self.captions = self.caption_file_data["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.caption_file_data)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        
        caption_vector = torch.tensor([self.vocab.stoi['<SOS>']] + self.vocab.numericalize(caption) + [self.vocab.stoi['<EOS>']])

        return image, caption_vector

class capscollate:
    def __init__(self, pad_idx, batch_first = True):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim = 0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first = self.batch_first, padding_value = self.pad_idx)
        return imgs, targets

def data_loader(root_dir, captions_file, transform = None, num_workers = 2, freq_threshold = 2, batch_size = 16):
    dataset = flickrdataset(root_dir, captions_file, transform, freq_threshold)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    return (dataset, DataLoader(dataset = dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True, collate_fn = capscollate(pad_idx, batch_first = True)))