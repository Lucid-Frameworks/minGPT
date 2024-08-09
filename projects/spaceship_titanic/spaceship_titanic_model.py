import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import get_column_embeddings

from IPython import embed


if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def evaluation(y, yhat):
    print('accuracy: ', accuracy_score(y, yhat))
    print('mean(y): ', np.mean(y))


def predict(model, dataloader, df):
    model.eval()

    yhat = []
    for input_ids, _ in dataloader:
        with torch.no_grad():
            output = model.generate(input_ids.to(device)).cpu().detach().numpy().tolist()
            yhat += np.array(output).squeeze().tolist()

    df["yhat"] = yhat
    return df


def feature_engineering(df):
    df_PassengerId = df["PassengerId"].str.split("_", expand=True)
    df["group"] = df_PassengerId[0].astype(int)
    df["number_in_group"] = df_PassengerId[1].astype(int)
    df["group_size"] = df.groupby("group")["number_in_group"].transform("count")
    df["single_group"] = np.where(df["group_size"] == 1, True, False)

    df_Name = df["Name"].str.split(" ", expand=True)
    df["first_name"] = df_Name[0]
    df["last_name"] = df_Name[1]

    df_Cabin = df["Cabin"].str.split("/", expand=True)
    df["deck"] = df_Cabin[0]
    df["num"] = df_Cabin[1]
    df["side"] = df_Cabin[2]
    df["cabin_size"] = df.groupby("Cabin")["num"].transform("count")
    df["single_cabin"] = np.where(df["cabin_size"] == 1, True, False)

    return df


def main(pretrained):
    np.random.seed(666)
    torch.manual_seed(42)

    # use data from Kaggle competition https://www.kaggle.com/competitions/spaceship-titanic
    df_train_full = pd.read_csv("train.csv")

    df_train_full = feature_engineering(df_train_full)

    categorical_features = [
        "single_group",
        "HomePlanet",
        "CryoSleep",
        "deck",
        "num",
        "side",
        "single_cabin",
        "Destination",
        "VIP",
    ]
    numerical_features = [
        "number_in_group",
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck"
    ]

    features = categorical_features + numerical_features

    df_train_full = df_train_full.fillna(-999)

    validation_groups = np.random.randint(0, len(df_train_full["group"].unique()), size=1000)
    df_val = df_train_full.loc[df_train_full["group"].isin(validation_groups)]
    df_train = df_train_full.loc[~df_train_full["group"].isin(validation_groups)]

    features_embeds_train = get_column_embeddings(df_train, "spaceship titanic", categorical_features, numerical_features, number_of_cols=16)
    features_embeds_val = get_column_embeddings(df_val, "spaceship titanic", categorical_features, numerical_features, number_of_cols=16)

    max_length = len(features) + 1

    train_dataset = TensorDataset(
        features_embeds_train, 
        torch.tensor(df_train["Transported"].tolist(), dtype=torch.long)
        )

    val_dataset = TensorDataset(
        features_embeds_val, 
        torch.tensor(df_val["Transported"].tolist(), dtype=torch.long)
        )

    # tabGPT model
    if pretrained:
        model = tabGPT.from_pretrained('gpt2', 2)
    else:
        model_config = tabGPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        # model_config.model_type = 'gpt2'
        model_config.vocab_size = 50257 # openai's model vocabulary
        model_config.block_size = max_length # 1024 is openai's model block_size
        model_config.n_output_nodes = 2
        model = tabGPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 10000
    train_config.epochs = 50
    train_config.num_workers = 0
    train_config.batch_size = 32
    trainer = Trainer(train_config, model, train_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

    # inference
    df_train = predict(model, DataLoader(train_dataset, batch_size=32), df_train)
    evaluation(df_train["Transported"], df_train["yhat"])

    df_val = predict(model, DataLoader(val_dataset, batch_size=32), df_val)
    evaluation(df_val["Transported"], df_val["yhat"])

    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    main(args.pretrained)
