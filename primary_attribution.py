from comet_ml import Experiment
import torch
import typer
from tqdm import tqdm
from typing import Optional
from preprocess import load_knn_data
from models import LinearProbe

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(experiment, args, model, train_loader):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    model.train()

    with experiment.train():
      for _ in range(args['epochs']):
          for article, label in tqdm(train_loader):
              article, label = article.to(device), label.to(device).squeeze()

              optimizer.zero_grad()

              probs = model(article)
              loss = loss_fn(probs, label)
              loss.backward()
              optimizer.step()

def test(experiment, model, test_loader):
    model.eval()
    with experiment.test():
        with torch.no_grad():
            cumulative_accuracy = 0
            num_batches = 0
            for article, label in tqdm(test_loader):
                article, label = article.to(device), label.to(device).squeeze()
                probs = model(article)
                classifications = probs.argmax(dim=1)
                corrects = (classifications == label)
                cumulative_accuracy += corrects.sum().float() / float(label.size(0))
                num_batches += 1
            accuracy = cumulative_accuracy / num_batches
            experiment.log_metric('Accuracy', accuracy)

def main(batch_size: Optional[int] = typer.Option(16, help='Input batch size for training (default: 64).'), 
    epochs: Optional[int] = typer.Option(1, help='Number of epochs to train (default: 10).'), 
    lr: Optional[float] = typer.Option(2e-5, help='Learning rate (default: 0.1).'), 
    seed: Optional[int] = typer.Option(1, help='Random seed (default: 1).'),
    model_name: Optional[str] = typer.Option('microsoft/deberta-base', help='Name of the transformer model (hugging face)'),
    save_model: Optional[bool] = typer.Option(True, help='For saving the current model.'),
    output_file: Optional[str] = typer.Option('probes/model.pt', help='The name of output file.'),
    model_file: Optional[str] = typer.Option('models/triplet_nf_1234_4e_10a.pt', help='The location of the saved model to probe.')):
    args = {
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'seed': seed,
        'save_model': save_model,
    }
    torch.manual_seed(seed)
    experiment = Experiment()
    train_loader, test_loader = load_knn_data(model_name, batch_size=batch_size)

    model = LinearProbe(model_file, model_name, embed_size=768).to(device)

    train(experiment, args, model, train_loader)

    if save_model:
        torch.save(model.state_dict(), output_file)

    test(experiment, model, test_loader)

if __name__=="__main__":
    main(typer.run(main))
    

