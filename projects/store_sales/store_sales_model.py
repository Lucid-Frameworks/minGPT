import sys

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
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
        ts = df.groupby(['date'])[['sales', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['date'])['sales'].sum().reset_index()
    plt.figure()
    ts.index = ts['date']
    ts['sales'].plot(style='r', label="sales")
    if include_preds:
        ts['yhat'].plot(style='b-.', label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("ts_{}.png".format(suffix))
    plt.clf()


def backtransform(df):
    df["yhat"] = np.exp(df["yhat"]) - 1
    return df


def evaluation(y, yhat):
    print('RMSLE: ', root_mean_squared_log_error(y, yhat))
    print('mean(y): ', np.mean(y))


def data_preparation(df):
    df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek

    df["sales_transformed"] = np.log(1 + df["sales"])

    def ewma_prediction(df, group_cols, col, alpha, horizon, suffix=''):
        df.sort_values(["date"], inplace=True)
        df_grouped = df.groupby(group_cols, group_keys=False)
        df["ewma_{}".format(col + suffix)] = df_grouped[col].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
        return df

    ewma_groups = ["store_nbr", "family", "dayofweek"]
    df = ewma_prediction(df, ewma_groups, "sales_transformed", 0.15, 1, suffix="_week")

    return df


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


def main(args):
    np.random.seed(666)
    torch.manual_seed(42)

    # use data from Kaggle competition https://www.kaggle.com/competitions/store-sales-time-series-forecasting
    df_train_full = pd.read_csv("train.csv")
    df_train_full = df_train_full[~np.isin(df_train_full["date"], ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"])]
    df_train_full = df_train_full[(df_train_full["date"] < "2016-04-16") | (df_train_full["date"] > "2016-05-01")]

    df_oil = pd.read_csv("oil.csv")
    df_train_full = df_train_full.merge(df_oil, on="date", how="left")

    df_train_full = data_preparation(df_train_full)

    # take just a small data set for testing
    df_train_full = df_train_full[(df_train_full["date"] >= "2017-05-01") & (df_train_full["store_nbr"].isin([1, 2, 3])) & (df_train_full["family"].isin(["LIQUOR,WINE,BEER", "EGGS", "MEATS"]))].reset_index()

    df_train_full.rename(columns={
        "store_nbr": "store",
        "family": "product group",
        "dayofweek": "weekday",
        "onpromotion": "promotion",
        "dcoilwtico": "oil price",
        "ewma_sales_transformed_week": "past sales",
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

    features_embeds_train = get_column_embeddings(df_train, "store sales", categorical_features, numerical_features, number_of_cols=6)
    features_embeds_val = get_column_embeddings(df_val, "store sales", categorical_features, numerical_features, number_of_cols=6)

    max_length = len(features) + 1

    train_dataset = TensorDataset(
        features_embeds_train, 
        torch.tensor(df_train["sales_transformed"].tolist(), dtype=torch.float32)
        )

    val_dataset = TensorDataset(
        features_embeds_val, 
        torch.tensor(df_val["sales_transformed"].tolist(), dtype=torch.float32)
        )

    # tabGPT model
    if args and args[0] == "--pretrained":
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
    evaluation(df_train["sales"], df_train["yhat"])
    plot_timeseries(df_train, "train", True)
    for pg in df_train["product group"].unique():
        plot_timeseries(df_train[df_train["product group"] == pg], pg + "_train", True)

    df_val = predict(model, DataLoader(val_dataset, batch_size=32), df_val)
    evaluation(df_val["sales"], df_val["yhat"])
    plot_timeseries(df_val, "val", True)
    for pg in df_val["product group"].unique():
        plot_timeseries(df_val[df_val["product group"] == pg], pg + "_val", True)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
