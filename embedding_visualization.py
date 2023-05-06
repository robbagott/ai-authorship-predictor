from comet_ml import Experiment
import torch
import typer
from typing import Optional
from tqdm import tqdm
from models import DebertaBase, BertBase
from preprocess import load_knn_data

device = 'cuda' if torch.cuda.is_available() else "cpu"

def main(
    model_name: Optional[str] = typer.Option('microsoft/deberta-base', help='The model name for the model_file.'),
    model_file: Optional[str] = typer.Option('model.pt', help='File name for the trained embedding model.'),
    batch_size: Optional[int] = typer.Option(16, help='The batch size for data processing in the embedding and knn models.')): 

    experiment = Experiment(
        api_key='VqZyAIH3L7ui07e9oY8wo61f7',
        project_name='AI Authorship Predictor',
        workspace='ameyerow2'
    )

    # Note: 768 is the embed size of deberta base model.
    if (model_name.lower() == "microsoft/deberta-base"):
        model = DebertaBase(model_name, 768, freeze=True).to(device)
    else:
        model = BertBase(model_name, 768, freeze=True).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))

    # Generate embeddings for knn training set.
    train_loader, test_loader = load_knn_data(model_name, batch_size=batch_size)
    train_embeds = []
    train_labels = []
    with torch.no_grad():
        for _, (articles, targets) in enumerate(tqdm(train_loader)):
            articles = articles.to(device)
            train_embeds.append(model(articles))
            train_labels.append(targets)
    # Flatten lists
    train_embeds = torch.cat(train_embeds).detach().cpu().numpy()
    train_labels = torch.cat(train_labels).detach().cpu().numpy()
    train_labels = train_labels.flatten()

    experiment.log_embedding(train_embeds, train_labels, title='Train embeddings')

    # Generate embeddings for knn test set.
    test_embeds = []
    test_labels = []
    with torch.no_grad():
        for _, (articles, targets) in enumerate(tqdm(test_loader)):
            articles = articles.to(device)
            test_embeds.append(model(articles))
            test_labels.append(targets)
    # Flatten lists
    test_embeds = torch.cat(test_embeds).detach().cpu().numpy()
    test_labels = torch.cat(test_labels).detach().cpu().numpy()
    test_labels = test_labels.flatten()

    experiment.log_embedding(test_embeds, test_labels, title='Test embeddings')


if __name__=="__main__":
    main('microsoft/deberta-base', 'models/contrast_deberta_all.pt', 16)