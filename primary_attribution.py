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

def primary_attribution(model, model_name, test_loader, n_steps):
    model.eval()
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        confident_real_article_text = "praying. Dinner in the dining room is self-service, cafeteria-style at 8 p.m. and Francis has been known to microwave his own food if it's not warm enough. Before taking the elevator back upstairs, he will be sure to thank the Swiss Guard, Vatican gendarme and reception desk clerk on duty in the hotel lobby, and say good-night. He's in bed by 9 p.m., reads for an hour and is asleep — \"like a log\" — for the next six hours or so. VACATION ROUTINE: Unlike his predecessors, Francis has never used the papal summer retreat at Castel Gandolfo in the hills south of Rome, preferring to stay home and just lighten his schedule. During vacation, Francis says he wakes up later and does more reading for pleasure, listening to music and praying. HOBBIES: Francis is a lifelong fan of soccer and has kept his membership in his beloved San Lorenzo club (Member No. 88235N-0). But he doesn't watch it on TV — in"
        uncertain_real_article_text = "Controversial Cape York MP Billy Gordon will not face domestic violence charges in Queensland after an investigation could not find sufficient evidence. Mr Gordon, who was forced out of the Labor Party in March for failing to disclose his criminal past, said last month police had completed their investigation. Late on Thursday, the Queensland Police Service issued a statement confirming the investigation had been completed. \"The police conclusion of the comprehensive investigation was referred for legal opinion,\" police said. \"After careful consideration of the investigation and legal opinion, it was determined there was"
        uncertain_fake_article_text = "September is a crucial month for many students hoping to obtain financial aid for college. The Free Application for Federal Student Aid (FAFSA) opens on October 1, but to be considered for aid, students need to complete it as soon as possible since aid is distributed on a first come, first served basis. Here's a financial aid tip for September: start gathering all necessary documents needed to complete the FAFSA. This includes tax returns, bank statements, and any other financial information. It's also important to understand the different types"
        confident_fake_article_text = "William Riggs, a prominent figure in the community, has recently gained recognition for his contributions to local philanthropy. Riggs has dedicated much of his time and resources to various charitable organizations throughout the years, making a significant impact on the lives of many. Born and raised in the area, Riggs has always felt a strong connection to his community. After achieving great success in his career, he made it his mission to give back to those in need. He has worked with numerous organizations, including food banks, homeless shelters, and youth mentoring programs. Riggs is also a fervent supporter of education,"
       
        articles = [confident_real_article_text, uncertain_real_article_text, uncertain_fake_article_text, confident_fake_article_text]
        labels = [1, 1, 0, 0]

        visualization_data_records = []

        for article_text, author_class in zip(articles, labels):
            article = torch.tensor([tokenizer.encode(article_text)]).to(device)
            label = torch.tensor([[author_class]]).to(device)
    
            # article, label = next(iter(test_loader))
            # article, label = article.to(device), label.to(device)

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
                article, label, n_steps=n_steps, return_convergence_delta=True, target=0
            )
            attributions = attributions.sum(dim=2).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.cpu().detach().numpy()

            visualization_data_record = visualization.VisualizationDataRecord(
                attributions,
                pred_prob,
                pred_class,
                true_class,
                "fake",
                attributions.sum(),
                text,
                delta)
            visualization_data_records.append(visualization_data_record)

        html_object = visualization.visualize_text(visualization_data_records)
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
    model_file: Optional[str] = typer.Option('models/triplet_nf_1234_4e_10a.pt', help='The location of the saved model to probe.'),
    n_steps: Optional[int] = typer.Option(500, help='Number of steps to run primary attribution algorithm for.')):
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
        primary_attribution(model, model_name, test_loader, n_steps)
    else:
        experiment = Experiment()
        
        train_loader, test_loader = load_knn_data(model_name, batch_size=batch_size)
        train(experiment, args, model, train_loader)

        if save_model:
            torch.save(model.state_dict(), output_file)

        test(experiment, model, test_loader)

        _, test_loader = load_knn_data(model_name, batch_size=1)
        primary_attribution(model, model_name, test_loader, n_steps)

if __name__=="__main__":
    main(typer.run(main))
    

