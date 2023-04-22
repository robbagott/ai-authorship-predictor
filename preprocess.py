import torch

DATA_DIR = '../data/'

# TODO: Produce training, test data.
def load_data(split=(0.8, 0.2), batch_size=128):
    signal_data = load_signal_data()
    gpt_data = load_gpt_data()

# TODO: Load data from files.
def load_signal_data():
    pass

# TODO: Load data from file.
def load_gpt_data():
    pass

# TODO: Generate triplets (anchor_idx, positive_idx, negative_idx)
def generate_triplets():
    pass

class ArticleDataset(torch.Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
    
    def __get_item__(self, index):
        pass

    def __len__(self):
        pass


