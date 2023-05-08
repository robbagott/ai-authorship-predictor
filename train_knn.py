import os
from comet_ml import Experiment
import torch
import typer
from typing import Optional
from tqdm import tqdm
from models import DebertaBase, BertBase
from preprocess import load_knn_data
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
import numpy as np

device = 'cuda' if torch.cuda.is_available() else "cpu"

def main(
    model_name: Optional[str] = typer.Option('microsoft/deberta-base', help='The model name for the model_file.'),
    model_file: Optional[str] = typer.Option('model.pt', help='File name for the trained embedding model.'),
    output_file: Optional[str] = typer.Option('knns/knn.pt', help='File name for the trained knn model to save to.'),
    results_file: Optional[str] = typer.Option('results/knn.txt', help='File name for the accuracy results to save to.'),
    batch_size: Optional[int] = typer.Option(16, help='The batch size for data processing in the embedding and knn models.')): 

    print(model_file, output_file, results_file)

    # Note: 768 is the embed size of deberta base model.
    if (model_name.lower() == "microsoft/deberta-base"):
        model = DebertaBase(model_name, 768, freeze=True).to(device)
    else:
        model = BertBase(model_name, 768, freeze=True).to(device)
    model.load_state_dict(torch.load(model_file))

    # Generate embeddings for knn training set.
    train_loader, test_loader = load_knn_data(model_name, batch_size=batch_size)
    train_embeds = []
    train_labels = []
    with torch.no_grad():
        for _, (articles, targets) in enumerate(tqdm(train_loader)):
            articles = articles.to(device)
            train_embeds.append(model(articles))
            train_labels.append(targets)
    train_embeds = torch.cat(train_embeds).detach().cpu().numpy()
    train_labels = torch.cat(train_labels).detach().cpu().numpy()
    train_labels = train_labels.flatten()

    # Generate embeddings for knn test set.
    test_embeds = []
    test_labels = []
    with torch.no_grad():
        for _, (articles, targets) in enumerate(tqdm(test_loader)):
            articles = articles.to(device)
            test_embeds.append(model(articles))
            test_labels.append(targets)
    test_embeds = torch.cat(test_embeds).detach().cpu().numpy()
    test_labels = torch.cat(test_labels).detach().cpu().numpy()
    test_labels = test_labels.flatten()

    experiment = Experiment(
        api_key='VqZyAIH3L7ui07e9oY8wo61f7',
        project_name='AI Authorship Predictor',
        workspace='ameyerow2'
    )

    with experiment.test():
        # Run a sweep for k values.
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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dump(knn, output_file)

if __name__ == '__main__':
    typer.run(main)
