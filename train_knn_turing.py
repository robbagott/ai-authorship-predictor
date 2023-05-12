import os
from comet_ml import Experiment
import torch
import typer
from typing import Optional
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from models import DebertaBase, BertBase
from preprocess import load_turing_bench

device = 'cuda' if torch.cuda.is_available() else "cpu"

def test_knn(experiment, train_embeds, train_labels, test_embeds, test_labels, results_file):
    best_model, best_score = None, 0
    # Run a sweep for k values.
    with experiment.test():
        for new_k in range(1, 101, 2):
            # Create the KNN from training set embeddings.
            knn = KNeighborsClassifier(n_neighbors=new_k)
            knn.fit(train_embeds, train_labels)

            # Test the KNN with test data.
            score = knn.score(test_embeds, test_labels)
            experiment.log_metric(name="KNN Accuracy", value=score, step=new_k)

            # Calculate percentage correct in predictions.
            probs = knn.predict_proba(test_embeds)
            prob_correct = probs[np.arange(len(probs)), test_labels].mean()
            experiment.log_metric(name="KNN Mean Prediction Probability", value=prob_correct, step=new_k)

            print(f'K = {new_k}, Accuracy: {score} Mean Probability: {prob_correct}')

            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'a') as file:
                file.write(f'{new_k}, {score}, {prob_correct}\n')
            
            if score > best_score:
                best_model, best_score = knn, score
    return best_model

def train_knn(experiment, model, model_name, task, results_file, batch_size):
    # Generate embeddings for knn training set.
    train_loader, test_loader = load_turing_bench(model_name, task, size=(500, 150), batch_size=batch_size)
    train_articles = []
    train_embeds = []
    train_labels = []
    with torch.no_grad():
        for _, (articles, targets) in enumerate(tqdm(train_loader)):
            train_articles.append(articles)
            articles = articles.to(device)
            train_embeds.append(model(articles))
            train_labels.append(targets)
    train_embeds = torch.cat(train_embeds).detach().cpu().numpy()
    train_labels = torch.cat(train_labels).detach().cpu().numpy()
    train_labels = train_labels.flatten()
    train_articles = torch.cat(train_articles)

    # Generate embeddings for knn test set.
    test_articles = []
    test_embeds = []
    test_labels = []
    with torch.no_grad():
        for _, (articles, targets) in enumerate(tqdm(test_loader)):
            test_articles.append(articles)
            articles = articles.to(device)
            test_embeds.append(model(articles))
            test_labels.append(targets)
    test_embeds = torch.cat(test_embeds).detach().cpu().numpy()
    test_labels = torch.cat(test_labels).detach().cpu().numpy()
    test_labels = test_labels.flatten()
    test_articles = torch.cat(test_articles)

    test_knn(experiment, train_embeds, train_labels, test_embeds, test_labels, results_file)

def main(
    model_name: Optional[str] = typer.Option('microsoft/deberta-base', help='The model name for the model_file.'),
    model_file: Optional[str] = typer.Option('model.pt', help='File name for the trained embedding model.'),
    results_file: Optional[str] = typer.Option('results/knn.txt', help='File name for the accuracy results to save to.'),
    task: Optional[str] = typer.Option('TT_gpt3', help='TuringBench task name.'),
    batch_size: Optional[int] = typer.Option(16, help='The batch size for data processing in the embedding and knn models.')): 

    # Note: 768 is the embed size of deberta base model.
    if (model_name.lower() == "microsoft/deberta-base"):
        model = DebertaBase(model_name, 768, freeze=True).to(device)
    else:
        model = BertBase(model_name, 768, freeze=True).to(device)
    model.load_state_dict(torch.load(model_file))

    experiment = Experiment()

    train_knn(experiment, model, model_name, task, results_file, batch_size)

if __name__ == '__main__':
    typer.run(main)
