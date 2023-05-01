from typing import List
import torch
import random
import pandas as pd
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_data(model_name, batch_size=64):
    train_path = 'data/train.csv'
    train_dataset = ArticleDataset(train_path, model_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_path = 'data/test.csv'
    test_dataset = ArticleDataset(test_path, model_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

class ArticleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, model_name, chunk_length=256, max_len=512):
        self.df = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.triplets = []
        self.A = []
        self.P = []
        self.N = []

        for index in tqdm(self.df.index):
            for author_class in ["fake", "real"]:
                article = self.df.iloc[index][author_class]
                anchor, positive_same = self._tokenize(self._chunk_article(article, chunk_length))

                negative_article = self.df.iloc[index][self._flip(author_class)]
                negative_same_1, negative_same_2 = self._tokenize(self._chunk_article(negative_article, chunk_length))

                random_positive = self._random_article(author_class, index)
                positive_random_1, positive_random_2 = self._tokenize(self._chunk_article(random_positive, chunk_length))

                random_negative = self._random_article(author_class, index)
                negative_random_1, negative_random_2 = self._tokenize(self._chunk_article(random_negative, chunk_length))

                # Positive instance from same article, negative from same article
                self.A.append(torch.tensor(anchor))
                self.P.append(torch.tensor(positive_same))
                self.N.append(torch.tensor(negative_same_1))

                # Positive instance from same article, negative from random article
                self.A.append(torch.tensor(anchor))
                self.P.append(torch.tensor(positive_same))
                self.N.append(torch.tensor(negative_random_1))

                # Positive instance from random article, negative from same article
                self.A.append(torch.tensor(anchor))
                self.P.append(torch.tensor(positive_random_1))
                self.N.append(torch.tensor(negative_same_2))

                # Positive instance from random article, negative from random article
                # TODO: can repeat this step an arbitrary number of times to generate more data
                self.A.append(torch.tensor(anchor))
                self.P.append(torch.tensor(positive_random_2))
                self.N.append(torch.tensor(negative_random_2))
        
        self.A = pad_sequence(self.A, batch_first=True)[:, :max_len]
        self.P = pad_sequence(self.P, batch_first=True)[:, :max_len]
        self.N = pad_sequence(self.N, batch_first=True)[:, :max_len]

    def _tokenize(self, chunks: List[str]):
        return [self.tokenizer(chunk)["input_ids"] for chunk in chunks]
    
    def _chunk_article(self, article: str, chunk_length: int) -> List[str]:
        split_article = article.split(' ')
        anchor_chunk_length = min(len(split_article)//2, chunk_length)

        first_chunk_start = random.choice(range(len(split_article)-2*anchor_chunk_length+1))
        second_chunk_start = random.choice(range(first_chunk_start+anchor_chunk_length, len(split_article)-anchor_chunk_length+1))

        chunks = [
            ' '.join(split_article[first_chunk_start:first_chunk_start+anchor_chunk_length]), 
            ' '.join(split_article[second_chunk_start:second_chunk_start+anchor_chunk_length])
        ]
        random.shuffle(chunks)

        return chunks

    def _random_article(self, author_class: str, prohibited_index: int) -> str:
        while True:
            random_index = random.choice(self.df.index)
            if random_index != prohibited_index:
                break
        return self.df.iloc[random_index][author_class]
    
    def _flip(self, author_class):
        return "fake" if author_class == "real" else "real"
    
    def __getitem__(self, index):
        return (self.A[index], self.P[index], self.N[index])

    def __len__(self):
        return len(self.A)

if __name__ == '__main__':
    dataset = ArticleDataset('data/train.csv', 'microsoft/deberta-v3-xsmall')
    loader = DataLoader(dataset, batch_size=10)
    item = next(iter(loader))
    a, p, n = item

    print(a.shape)
    print(p.shape)
    print(n.shape)