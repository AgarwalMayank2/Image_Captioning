import pandas as pd
from collections import Counter

def get_vocab(captions):
    words_list = captions.apply(lambda x: " ".join(x)).str.cat(sep = ' ').split()
    word_counts = Counter(words_list)
    word_counts = sorted(word_counts, key = word_counts.get, reverse = True)

    vocab_size = len(word_counts)
    index_to_word = {index: word for index, word in enumerate(word_counts)}
    word_to_index = {word: index for index, word in enumerate(word_counts)}

    return vocab_size, index_to_word, word_to_index

class PreProcess:
    def __init__(self, location):
        super().__init__()
        data = pd.read_csv(location)

        data['cleaned_captions'] = data['caption'].apply(lambda x: ['<start>'] + [word.lower() for word in x.split()] + ['<end>'])

        max_seq_length = 0
        for caption in data['cleaned_captions']:
            if len(caption) > max_seq_length:
                max_seq_length = len(caption)

        data['cleaned_captions'] = data['cleaned_captions'].apply(lambda x: x + ['<pad>']*(max_seq_length - len(x)))

        self.vocab_size, self.itow, self.wtoi = get_vocab(data['cleaned_captions'])

        data['tokenized_captions'] = data['cleaned_captions'].apply(lambda x: [self.wtoi[word] for word in x])

        self.data = data