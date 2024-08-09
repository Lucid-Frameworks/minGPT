import sys

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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


def plot_timeseries(df, suffix, include_preds=False):
    if include_preds:
        ts = df.groupby(['date'])[['target', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['date'])['target'].sum().reset_index()
    plt.figure()
    ts.index = ts['date']
    ts['target'].plot(style='r', label="target")
    if include_preds:
        ts['yhat'].plot(style='b-.', label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("ts_{}.png".format(suffix))
    plt.clf()


def backtransform(df):
    df["yhat"] = np.exp(df["yhat"]) - 1
    df["target"] = np.exp(df["target"]) - 1
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


def get_data_store_sales():
    # use data from Kaggle competition https://www.kaggle.com/competitions/store-sales-time-series-forecasting
    df_train_full = pd.read_csv("train_store_sales.csv")
    df_train_full = df_train_full[~np.isin(df_train_full["date"], ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"])]
    df_train_full = df_train_full[(df_train_full["date"] < "2016-04-16") | (df_train_full["date"] > "2016-05-01")]

    df_oil = pd.read_csv("oil.csv")
    df_train_full = df_train_full.merge(df_oil, on="date", how="left")

    df_train_full['dayofweek'] = pd.to_datetime(df_train_full['date']).dt.dayofweek
    df_train_full["target"] = np.log(1 + df_train_full["sales"])

    def ewma_prediction(df, group_cols, col, alpha, horizon, suffix=''):
        df.sort_values(["date"], inplace=True)
        df_grouped = df.groupby(group_cols, group_keys=False)
        df["ewma_{}".format(col + suffix)] = df_grouped[col].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
        return df

    ewma_groups = ["store_nbr", "family", "dayofweek"]
    df_train_full = ewma_prediction(df_train_full, ewma_groups, "target", 0.15, 1, suffix="_week")

    # take just a small data set for testing
    df_train_full = df_train_full[(df_train_full["date"] >= "2017-05-01") & (df_train_full["store_nbr"].isin([1, 2, 3])) & (df_train_full["family"].isin(["LIQUOR,WINE,BEER", "EGGS", "MEATS"]))].reset_index()

    df_train_full.rename(columns={
        "store_nbr": "store",
        "family": "product group",
        "dayofweek": "weekday",
        "onpromotion": "promotion",
        "dcoilwtico": "oil price",
        "ewma_target_week": "past sales",
    }, inplace=True)
    categorical_features = [
        "store",
        "product group",
        "weekday",
    ]
    numerical_features = [
        "promotion",
        "oil price",
        "past sales",
    ]

    features = categorical_features + numerical_features

    df_train_full = df_train_full.fillna(-999)

    df_train = df_train_full[df_train_full["date"] <= "2017-07-30"].reset_index()
    df_val = df_train_full[df_train_full["date"] >= "2017-07-31"].reset_index()

    return df_train, df_val, features, categorical_features, numerical_features


def get_data_house_prices():
    # use data from Kaggle competition https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
    df_train_full = pd.read_csv("train_house_prices.csv")

    df_train_full["target"] = np.log(1 + df_train_full["SalePrice"])

    categorical_features = [
        "OverallQual",
        "ExterQual",
        "Neighborhood",
        "BsmtQual",
        "YearBuilt",
        "KitchenQual",
    ]
    numerical_features = [
        "GarageCars",
        "GrLivArea",
        "GarageArea",
        "TotalBsmtSF"            
    ]

    features = categorical_features + numerical_features

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    return df_train, df_val, features, categorical_features, numerical_features


def main(args):
    np.random.seed(666)
    torch.manual_seed(42)

    df_train_store_sales, df_val_store_sales, features_store_sales, categorical_features_store_sales, numerical_features_store_sales = get_data_store_sales()
    df_train_house_prices, df_val_house_prices, features_house_prices, categorical_features_house_prices, numerical_features_house_prices = get_data_house_prices()

    features = features_store_sales + features_house_prices

    features_embeds_train_store_sales = get_column_embeddings(df_train_store_sales, "store sales", categorical_features_store_sales, numerical_features_store_sales)
    features_embeds_train_house_prices = get_column_embeddings(df_train_house_prices, "house prices", categorical_features_house_prices, numerical_features_house_prices)

    features_embeds_train = torch.cat((features_embeds_train_store_sales, features_embeds_train_house_prices), dim=0)

    max_length = len(features) + 2

    targets_train = df_train_store_sales["target"].tolist() + df_train_house_prices["target"].tolist()

    train_dataset = TensorDataset(
        features_embeds_train, 
        torch.tensor(targets_train, dtype=torch.float32)
        )

    # tabGPT model
    if args and args[0] == "--pretrained":
        model = tabGPT.from_pretrained('gpt2', 1)
    else:
        model_config = tabGPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        # model_config.model_type = 'gpt2'
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
    features_embeds_val_store_sales = get_column_embeddings(df_val_store_sales, "store sales", categorical_features_store_sales, numerical_features_store_sales)
    features_embeds_val_house_prices = get_column_embeddings(df_val_house_prices, "house prices", categorical_features_house_prices, numerical_features_house_prices)

    val_dataset_store_sales = TensorDataset(
        features_embeds_val_store_sales, 
        torch.tensor(df_val_store_sales["target"].tolist(), dtype=torch.float32)
        )

    val_dataset_house_prices = TensorDataset(
        features_embeds_val_house_prices, 
        torch.tensor(df_val_house_prices["target"].tolist(), dtype=torch.float32)
        )

    df_val_store_sales = predict(model, DataLoader(val_dataset_store_sales, batch_size=32), df_val_store_sales)
    evaluation(df_val_store_sales["target"], df_val_store_sales["yhat"])
    plot_timeseries(df_val_store_sales, "val", True)

    df_val_house_prices = predict(model, DataLoader(val_dataset_house_prices, batch_size=32), df_val_house_prices)
    evaluation(df_val_house_prices["target"], df_val_house_prices["yhat"])

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
