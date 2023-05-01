import torch
import typer
from typing import Optional
import tqdm
from models import DebertaBase, BertBase
from preprocess import load_knn_data
import sklearn

device = 'cuda' if torch.cuda.is_available() else "cpu"

def main(
    model_type: Optional[str] = typer.Option('deberta-base', help='The model class for the model_file.'),
    model_file: Optional[str] = typer.Option('model.pt', help='File name for the trained embedding model.'),
    k: Optional[int] = typer.Option('5', help='Number of neighbors for KNN.')): 

    # Load the embedding model.
    if model_type.lower() == "deberta-base":
        model = DebertaBase('microsoft/deberta-base', 768, probe=True).to(device)
    if model_type.lower() == "bert-base":
        model = BertBase('microsoft/bert-base-cased', 768, probe=True).to(device)
    model.load_state_dict(torch.load(model_file))

    # Generate embeddings for knn training set.
    train_loader, test_loader = load_knn_data()
    embeds = []
    labels = []
    with torch.no_grad():
        for _, (articles, targets) in enumerate(tqdm(train_loader)):
            articles, targets = articles.to(device), targets.to(device)
            embeds.append(model(articles))
            labels.append(targets)
    embeds = torch.cat(embeds).detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()
    
    # Create the KNN.
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeds, labels)

    # Test the KNN with some test data.
    correct = 0
    total = 0
    for _, (articles, targets) in enumerate(tqdm(train_loader)):
        pred = knn.predict(articles)
        correct += (pred == targets).sum()
        total += articles.shape[0]

    print('Accuracy: ' + correct / total)

if __name__ == '__main__':
    typer.run(main)
