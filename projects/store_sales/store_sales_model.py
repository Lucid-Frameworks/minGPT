import sys

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.bpe import get_encoder

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


def padding(input_ids, max_length):
    for i, ids in enumerate(input_ids):
        pad_list = [666] * (max_length - len(ids)) # negative values throw error in embeddings
        input_ids[i] = ids + pad_list
    return input_ids


def encode_text(df, enc):
    input_ids = []
    for text in df["text"].tolist():
        input_ids.append(enc.encode(text))
    return input_ids


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
    features = [
        "store",
        "product group",
        "weekday",
        "promotion",
        # "oil price",
        # "past sales",
    ]

    df_train_full["text"] = ""
    for col in features:
        df_train_full["text"] += col + ": " + df_train_full[col].astype(str) + "\n"

    df_train = df_train_full[df_train_full["date"] <= "2017-07-30"].reset_index()
    df_val = df_train_full[df_train_full["date"] >= "2017-07-31"].reset_index()

    enc = get_encoder()
    input_ids_train = encode_text(df_train, enc)
    input_ids_val = encode_text(df_val, enc)

    max_length = 0
    for ids in input_ids_train:
        len_ids = len(ids)
        if len_ids > max_length:
            max_length = len_ids

    input_ids_train = padding(input_ids_train, max_length)
    input_ids_val = padding(input_ids_val, max_length)

    train_dataset = TensorDataset(
        torch.tensor(input_ids_train), 
        torch.tensor(df_train["sales_transformed"].tolist(), dtype=torch.float32)
        )
    val_dataset = TensorDataset(
        torch.tensor(input_ids_val), 
        torch.tensor(df_val["sales_transformed"].tolist(), dtype=torch.float32)
        )

    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    # model_config.model_type = 'gpt2'
    # model_config.vocab_size = len(np.unique(input_ids_train))
    model_config.vocab_size = 50257 # openai's model vocabulary
    model_config.block_size = max_length
    model = GPT(model_config)

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
