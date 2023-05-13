# AI Authorship Prediction

  Distinguishing machine-generated text from human text has never been more important. We propose a purely contrastive learning approach to solving the problem of AI authorship attribution. Our results show that fine-tuning a pre-trained DeBERTa model with a margin-based triplet loss results in a highly separated embedding space, and achieves approximately 0.95 accuracy in distinguishing ChatGPT-generated news articles from their human-authored counterparts. We also show that this detection ability generalizes to other chat bots by using the TuringBench dataset, which the embedding space was never trained on.

  # Environment Setup

  This project was created using python 3.10.4, it is recommended that you install this version for compatibility. 

  Before installing python packages, first create aand activate a python virutal environment.

```
python -m venv ai-authorship
```
To active on windows:
```
.\ai-authorship\Scripts\activate
```
To activate on Linux or Mac:
```
source ai-authorship/bin/activate
```

Then install the required python packages using pip.
```
pip install -r requirements.txt
```

# Logging via Comet.ml
In this project we use [Comet.ml](https://www.comet.com/) for training and testing metrics, and embedding visualizations. Before running code, first create an account and a new project which your experiments will be logged to. Then then store your API key (found in your account settings), project workspace, and project name into `.comet.config` (note that no quotation marks are required around strings in this config).

# Data

We have generated a train and test set of fake-real article pairs using ChatGPT to reproduce its own version of real articles in the [Signal Media 1M dataset](http://ceur-ws.org/Vol-1568/paper8.pdf). These can be found at `data/train.csv` and `data/test.csv`.

To generate your own additional data with this technique, you can use `data_creation.py` for reference. This will require downloading the Signal Media dataset and entering your OpenAI API key in the associated field in the python file. For reference, generating 500 training articles and 150 test articles cost a little of $1.

# Model Weights

Our best performing model was trained with triplet loss, batch size set to 16, and alpha set to 10. Our weights can be downloaded [here](https://drive.google.com/file/d/1tLYQgj-WtaWC0j1T8lh7Gc00HVXEau7N/view?usp=sharing).

# Training

A training job can be started from the root directory with the following command after activating your virtual environment.

```
python train.py
```

By default, this is run with the DeBERTa weights frozen out of concern for the memory required to do fine-tuning. To fine tune it is recommended to have a GPU with at least 12GB of VRAM. Training for our saved weights was done on a RTX 3090. To unfreeze the weights run training with the following flag.

```
python train.py --no-freeze
```

A full list of arguments and explanations to the train script can be found using:

```
python train.py --help
```

# KNN Classification

To run KNN on classification with a pretrained model,  you can use the `train_knn.py` script. Note that this is run as part of training automatically. This will require passing in your pretrained model with the following flag:

```
python train_knn.py --model-file PATH/TO/MODEL
```

# Primary Attribution

To visualize the primary attribution for a trained model we train a simple linear probe on top of our frozen contrastive learning model to convert this our pipeline to a normal classification problem. To train a linear probe on a pretrained model use the following:

```
python primary_attribution.py --model-file PATH/TO/MODEL --no-load-model
```

Once the linear probe is trained, you can rerun the script without retraining using the saved probe.

```
python primary_attribution.py --load-model
```

# Results

[Embedding visualizations for our best performing model](https://www.comet.com/projector/?config=%2Fapi%2Fasset%2Fdownload%3FassetId%3D328daaf39d4f43c9af3f683aae0d23f8%26experimentKey%3Dd7ace6557d824854b476487f16cf60df)

[Embedding visualization with article labels](https://www.comet.com/projector/?config=%2Fapi%2Fasset%2Fdownload%3FassetId%3Dd92772739fc7456c8d34eb0663dde492%26experimentKey%3Dd7ace6557d824854b476487f16cf60df)