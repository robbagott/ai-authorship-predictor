import torch
import typer
from typing import Optional
from tqdm import tqdm
from preprocess import load_knn_data
import pandas as pd
from matplotlib import pyplot as plt
from transformers import AutoTokenizer


def main(model_name: Optional[str] = typer.Option('microsoft/deberta-base', help='The model name for the model_file.')): 

    # Generate embeddings for knn training set.
    train_loader, _ = load_knn_data(model_name, batch_size=1)

    label_1_data = {}
    label_0_data = {}
    key_1 = 0
    key_0 = 0
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for i, (articles, targets) in enumerate(tqdm(train_loader)):
        label = targets.item()
        if label == 0:
            label_0_data[i] = articles.tolist()[0]
            key_0 = i
        else:
            label_1_data[i] = articles.tolist()[0]
            key_1 = i
    
    print(tokenizer.decode(label_0_data[key_0]))
    print(tokenizer.decode(label_1_data[key_1]))

    df = pd.DataFrame.from_dict(label_1_data)
    index = df.index

    pd.value_counts(df.loc[index[1]]).plot.barh()
    plt.show()

    df = pd.DataFrame.from_dict(label_0_data)
    index = df.index

    pd.value_counts(df.loc[index[1]]).plot.barh()
    plt.show()


if __name__=="__main__":
    main('microsoft/deberta-base')