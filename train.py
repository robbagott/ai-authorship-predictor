from comet_ml import Experiment
from rich import print
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import typer
from typing import Optional

from models import DebertaBase, BertBase
from preprocess import load_data
from losses import TripletLoss, triplet_acc, ContrastLoss, contrast_acc, NcaHnLoss, MarginHnLoss, MixedLoss
from test import test

device = 'cuda' if torch.cuda.is_available() else "cpu"

# Train/test code courtesy of pytorch examples repo. (https://github.com/pytorch/examples/blob/main/mnist/main.py#L12)
def train(experiment, args, model, device, train_loader, optimizer, loss_fn):
    """
    :param args command line arguments
    :param model model to be trained
    :param device either cuda or CPU depending on availability
    :param train_loader a Pytorch Dataloader of the trainset and labels
    :param optimizer nn.optimizer (Adam)
    :param loss_fn Loss function to train model to.
    :param writer The tensorboard SummaryWriter to log data on.
    """
    model.train()

    with experiment.train():
      for _ in range(args['epochs']):
          for a, p, n in tqdm(train_loader):
              a, p, n = a.to(device), p.to(device), n.to(device)
              optimizer.zero_grad()

              a_embed = model(a)
              p_embed = model(p)
              n_embed = model(n)
              loss = loss_fn(a_embed, p_embed, n_embed)
              loss.backward()
              optimizer.step()

def main(
    batch_size: Optional[int] = typer.Option(16, help='Input batch size for training (default: 64).'), 
    epochs: Optional[int] = typer.Option(10, help='Number of epochs to train (default: 10).'), 
    lr: Optional[float] = typer.Option(2e-5, help='Learning rate (default: 0.1).'), 
    seed: Optional[int] = typer.Option(1, help='Random seed (default: 1).'),
    log_interval: Optional[int] = typer.Option(10, help='how many batches to wait before logging training status (default: 10).'),
    model_name: Optional[str] = typer.Option('microsoft/deberta-base', help='Name of the transformer model (hugging face)'),
    save_model: Optional[bool] = typer.Option(True, help='For saving the current model.'),
    output_file: Optional[str] = typer.Option('model.pt', help='The name of output file.'),
    data_option: Optional[str] = typer.Option('1234', help='"1" for same content triplets. "2" for easy negatives. "3" for hard negatives. "4" for random article positives and negatives.'),
    loss: Optional[str] = typer.Option('triplet', help='"Triplet" for triplet loss. "Contrast" for contrast loss.'),
    alpha: Optional[float] = typer.Option(1, help='Margin value for triplet loss.'),
    temp: Optional[float] = typer.Option(0.1, help='Temperature value for constrast loss.'),
    freeze: Optional[bool] = typer.Option(True, help='True if the base embedding model should be frozen.')):
    args = {
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'seed': seed,
        'log_interval': log_interval,
        'save_model': save_model,
    }
    torch.manual_seed(seed)

    experiment = Experiment(
        api_key='VqZyAIH3L7ui07e9oY8wo61f7',
        project_name='AI Authorship Predictor',
        workspace='ameyerow2'
    )

    train_loader, test_loader = load_data(model_name, batch_size, data_option)

    # Note: 768 is the embed size of deberta base model.
    if (model_name.lower() == "microsoft/deberta-base"):
      model = DebertaBase(model_name, 768, freeze=freeze).to(device)
    else:
        model = BertBase(model_name, 768, freeze=freeze).to(device)
    
    if (loss.lower() == "contrast"):
      loss_fn = ContrastLoss(temp)
      acc_fn = contrast_acc()
    elif (loss.lower() == "triplet"):
      loss_fn = TripletLoss(alpha)
      acc_fn = triplet_acc()
    elif (loss.lower() == "ncahn"):
      loss_fn = NcaHnLoss()
      acc_fn = contrast_acc()
    elif (loss.lower() == "marginhn"):
      loss_fn = MarginHnLoss(alpha)
      acc_fn = triplet_acc()
    else:
       loss_fn = MixedLoss(temp)
       acc_fn = contrast_acc()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train(experiment, args, model, device, train_loader, optimizer, loss_fn)

    if save_model:
        torch.save(model.state_dict(), output_file)

    test(experiment, model, device, test_loader, loss_fn, acc_fn, verbose=True)

if __name__ == '__main__':
    typer.run(main)
