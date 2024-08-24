import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

from tabgpt.model import tabGPT
from tabgpt.trainer import Trainer
from tabgpt.col_embed import get_column_embeddings
from projects.house_prices.construct_dataset import construct_text

from IPython import embed


if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def backtransform(df):
    df["yhat"] = np.exp(df["yhat"]) - 1
    return df


def evaluation(y, yhat):
    print('RMSLE: ', root_mean_squared_log_error(y, yhat))
    print('mean(y): ', np.mean(y))


def predict(model, dataloader, df):
    model.eval()

    yhat = []
    for input_ids, _ in dataloader:
        with torch.no_grad():
            output = model.generate(input_ids.to(device)).cpu().detach().numpy().tolist()
            yhat += np.array(output).squeeze().tolist()

    df["yhat"] = yhat
    df["yhat"] = np.clip(df["yhat"], 0, None)
    df = backtransform(df)
    return df


def main(pretrained, enrich):
    np.random.seed(666)
    torch.manual_seed(42)

    if enrich:
        df_train_full = construct_text()
        categorical_features = [
            "Overall material and finish of the house",
            "Quality of the material on the exterior",
            "Physical locations within Ames city limits",
            "Height of the basement",
            "Original construction date",
            "Kitchen quality",
        ]
        numerical_features = [
            "Size of garage in car capacity",
            "Above grade (ground) living area square feet",
            "Size of garage in square feet",
            "Total square feet of basement area"            
        ]
    else:
        # use data from Kaggle competition https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
        df_train_full = pd.read_csv("train.csv")
        # categorical_features = [
        #     "OverallQual",
        #     "ExterQual",
        #     "Neighborhood",
        #     "BsmtQual",
        #     "YearBuilt",
        #     "KitchenQual",
        # ]
        # numerical_features = [
        #     "GarageCars",
        #     "GrLivArea",
        #     "GarageArea",
        #     "TotalBsmtSF"            
        # ]
        categorical_features = []
        numerical_features = []
        for col in df_train_full.drop(columns=["Id", "SalePrice"]).columns:
            if len(df_train_full[col].unique()) < 2:
                print(f"exclude column {col} with only one unique value")
            elif df_train_full[col].dtype == 'O':
                categorical_features.append(col)
            else:
                numerical_features.append(col)

    features = categorical_features + numerical_features

    df_train_full = df_train_full.fillna(-999)

    df_train_full["SalePrice_transformed"] = np.log(1 + df_train_full["SalePrice"])

    df_train_full = df_train_full[['Id', 'SalePrice', 'SalePrice_transformed'] + features]

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    features_embeds_train = get_column_embeddings(df_train, "house prices", categorical_features, numerical_features, number_of_cols=len(features))

    train_dataset = TensorDataset(
        features_embeds_train, 
        torch.tensor(df_train["SalePrice_transformed"].tolist(), dtype=torch.float32)
        )

    max_length = len(features) + 1

    # tabGPT model
    if pretrained:
        model = tabGPT.from_pretrained('gpt2', 1)
    else:
        model_config = tabGPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        model_config.vocab_size = 50257 # openai's model vocabulary
        model_config.block_size = max_length # 1024 is openai's model block_size
        model_config.n_output_nodes = 1
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
    evaluation(df_train["SalePrice"], df_train["yhat"])

    features_embeds_val = get_column_embeddings(df_val, "house prices", categorical_features, numerical_features, number_of_cols=len(features))
    val_dataset = TensorDataset(
        features_embeds_val, 
        torch.tensor(df_val["SalePrice_transformed"].tolist(), dtype=torch.float32)
        )
    df_val = predict(model, DataLoader(val_dataset, batch_size=32), df_val)
    evaluation(df_val["SalePrice"], df_val["yhat"])

    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--enrich", action="store_true")
    args = parser.parse_args()
    main(args.pretrained, args.enrich)
