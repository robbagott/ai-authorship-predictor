from comet_ml import Experiment
import torch
import typer
from tqdm import tqdm
from typing import Optional
from preprocess import load_knn_data
from models import LinearProbe
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from transformers import AutoTokenizer

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

def primary_attribution(model, model_name, test_loader):
    model.eval()
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        article, label = next(iter(test_loader))
        article, label = article.to(device), label.to(device)

        output = model(article)
        pred_prob = torch.max(output)
        pred_ind = torch.argmax(output)
        pred_class = "fake" if pred_ind == 0 else "real"
        true_class = "fake" if label == 0 else " real"
        text = tokenizer.decode(article[0]).split(' ')

        lig = LayerIntegratedGradients(
            model,
            model.deberta.deberta.embeddings.word_embeddings,
        )
        attributions, delta = lig.attribute(
            article, label, n_steps=5, return_convergence_delta=True, target=0
        )
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        html_object = visualization.visualize_text([
            visualization.VisualizationDataRecord(
                attributions,
                pred_prob,
                pred_class,
                true_class,
                "fake",
                attributions.sum(),
                text,
                delta
            )
        ])
        with open("data.html", "w") as file:
            file.write(html_object.data)

def main(batch_size: Optional[int] = typer.Option(16, help='Input batch size for training (default: 64).'), 
    epochs: Optional[int] = typer.Option(1, help='Number of epochs to train (default: 10).'), 
    lr: Optional[float] = typer.Option(2e-5, help='Learning rate (default: 0.1).'), 
    seed: Optional[int] = typer.Option(1, help='Random seed (default: 1).'),
    load_model: Optional[bool] = typer.Option(True, help='Load a pretrained Linear Probe store at output_file location'),
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

    model = LinearProbe(model_file, model_name, embed_size=768).to(device)

    if load_model:
        train_loader, test_loader = load_knn_data(model_name, batch_size=1)
        model.load_state_dict(torch.load(output_file, map_location=device))
        primary_attribution(model, model_name, test_loader)
    else:
        experiment = Experiment()
        
        train_loader, test_loader = load_knn_data(model_name, batch_size=batch_size)
        train(experiment, args, model, train_loader)

        if save_model:
            torch.save(model.state_dict(), output_file)

        test(experiment, model, test_loader)

        _, test_loader = load_knn_data(model_name, batch_size=1)
        primary_attribution(model, model_name, test_loader)




if __name__=="__main__":
    main(typer.run(main))
    

