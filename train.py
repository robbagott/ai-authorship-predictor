from rich import print
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import typer
from typing import Optional

from models import DebertaBase
from preprocess import load_data
from losses import TripletLoss
from losses import triplet_acc

device = 'cuda' if torch.cuda.is_available() else "cpu"

# Train/test code courtesy of pytorch examples repo. (https://github.com/pytorch/examples/blob/main/mnist/main.py#L12)
def train(args, model, device, train_loader, optimizer, loss_fn, writer):
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
    n_batches = 0

    for _ in range(args['epochs']):
        for batch_idx, (a, p, n) in enumerate(tqdm(train_loader)):
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()

            a_embed = model(a)
            p_embed = model(p)
            n_embed = model(n)
            loss = loss_fn(a_embed, p_embed, n_embed)
            loss.backward()
            optimizer.step()

            n_batches += 1
            if batch_idx % args['log_interval'] == 0:
                writer.add_scalar('Loss/train', loss, n_batches)

# Tests a model with triplet accuracy.
def test(model, device, test_loader, loss_fn, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for a, p, n in test_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            a_embed = model(a)
            p_embed = model(p)
            n_embed = model(n)
            loss += torch.sum(loss_fn(a_embed, p_embed, n_embed)).item()
            acc += torch.sum(triplet_acc(a_embed, p_embed, n_embed)).item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Triplet accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return test_loss, test_acc

def main(
    batch_size: Optional[int] = typer.Option(64, help='Input batch size for training (default: 64).'), 
    test_batch_size: Optional[int] = typer.Option(1000, help='Input batch size for testing (default: 1000).'), 
    epochs: Optional[int] = typer.Option(10, help='Number of epochs to train (default: 10).'), 
    lr: Optional[float] = typer.Option(0.1, help='Learning rate (default: 0.1).'), 
    no_cuda: Optional[bool] = typer.Option(False, help='Disables CUDA training (default: False).'), 
    no_mps: Optional[bool] = typer.Option(False, help='Disables macOS GPU training (default: False).'), 
    seed: Optional[int] = typer.Option(1, help='Random seed (default: 1).'),
    log_interval: Optional[int] = typer.Option(10, help='how many batches to wait before logging training status (default: 10).'),
    save_model: Optional[bool] = typer.Option(False, help='For saving the current model.'),
    alpha: Optional[float] = typer.Option(1, help='Margin value for triplet loss.')):
    args = {
        'batch_size': batch_size,
        'test_batch_size': test_batch_size,
        'epochs': epochs,
        'lr': lr,
        'no_cuda': no_cuda,
        'no_mps': no_mps,
        'seed': seed,
        'log_interval': log_interval,
        'save_model': save_model,
    }
    torch.manual_seed(seed)

    train_loader, test_loader = load_data()

    # Note: 384 is the embed size of xsmall deberta model.
    model = DebertaBase('microsoft/deberta-v3-xsmall', 384, probe=True).to(device)
    loss_fn = TripletLoss(alpha)
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(args, model, device, train_loader, test_loader, optimizer, loss_fn, writer)
    test(model, device, test_loader, loss_fn, verbose=True)

    if save_model:
        torch.save(model.state_dict(), "model.pt")

if __name__ == '__main__':
    typer.run(main)
