from comet_ml import Experiment
import torch
import typer
from typing import Optional

from losses import TripletLoss, triplet_acc, ContrastLoss, contrast_acc, NcaHnLoss, MarginHnLoss, MixedLoss
from models import DebertaBase, BertBase
from preprocess import load_data

device = 'cuda' if torch.cuda.is_available() else "cpu"

# Tests a model with triplet accuracy.
def test(experiment, model, device, test_loader, loss_fn, acc_fn, verbose=False):
    model.eval()
    with experiment.test():
      test_loss = 0
      correct = 0
      test_acc = 0
      with torch.no_grad():
          for a, p, n in test_loader:
              a, p, n = a.to(device), p.to(device), n.to(device)
              a_embed = model(a)
              p_embed = model(p)
              n_embed = model(n)
              test_loss += torch.sum(loss_fn(a_embed, p_embed, n_embed)).item()
              correct += torch.sum(acc_fn(a_embed, p_embed, n_embed)).item()

      test_loss /= len(test_loader.dataset)
      test_acc = correct / len(test_loader.dataset)
      if verbose:
          print('\nTest set: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
              test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))
      experiment.log_metric('Test Loss', test_loss)
      experiment.log_metric('Test Accuracy', test_acc)
      return test_loss, test_acc

def main(
    batch_size: Optional[int] = typer.Option(64, help='Input batch size for training (default: 64).'), 
    model_file: Optional[str] = typer.Option('model.pt', help='Path to model file to load for testing.'),
    model_name: Optional[str] = typer.Option('microsoft/deberta-base', help='Name of the transformer model (hugging face)'),
    loss: Optional[str] = typer.Option('triplet', help='"Triplet" for triplet loss. "Contrast" for contrast loss.'),
    alpha: Optional[float] = typer.Option(1, help='Margin value for triplet loss.'),
    temp: Optional[float] = typer.Option(0.1, help='Temperature value for constrast loss.')):

    experiment = Experiment()

    _, test_loader = load_data(model_name, batch_size)

    # Note: 768 is the embed size of deberta base model.
    if (model_name.lower() == "microsoft/deberta-base"):
      model = DebertaBase(model_name, 768, freeze=True).to(device)
    else:
        model = BertBase(model_name, 768, freeze=True).to(device)
    model.load_state_dict(torch.load(model_file))

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

    test(experiment, model, device, test_loader, loss_fn, acc_fn, verbose=True)

if __name__ == '__main__':
    typer.run(main)