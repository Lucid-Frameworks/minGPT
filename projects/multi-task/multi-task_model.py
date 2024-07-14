import sys

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
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


def evaluation(df):
    print('data set: store sales')
    mask = (df["dataset"] == "store sales")
    print('RMSLE: ', root_mean_squared_log_error(df[mask]["target"], df[mask]["yhat"]))
    print('mean(y): ', np.mean(df[mask]["target"]))

    print('data set: house prices')
    mask = (df["dataset"] == "house prices")
    print('RMSLE: ', root_mean_squared_log_error(df[mask]["target"], df[mask]["yhat"]))
    print('mean(y): ', np.mean(df[mask]["target"]))


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


def get_data_store_sales():
    # use data from Kaggle competition https://www.kaggle.com/competitions/store-sales-time-series-forecasting
    df_train_full = pd.read_csv("train_store_sales.csv")
    df_train_full = df_train_full[~np.isin(df_train_full["date"], ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"])]
    df_train_full = df_train_full[(df_train_full["date"] < "2016-04-16") | (df_train_full["date"] > "2016-05-01")]

    df_train_full['dayofweek'] = pd.to_datetime(df_train_full['date']).dt.dayofweek
    df_train_full["target"] = np.log(1 + df_train_full["sales"])

    # take just a small data set for testing
    df_train_full = df_train_full[(df_train_full["date"] >= "2017-05-01") & (df_train_full["store_nbr"].isin([1, 2, 3])) & (df_train_full["family"].isin(["LIQUOR,WINE,BEER", "EGGS", "MEATS"]))].reset_index()

    df_train_full.rename(columns={
        "store_nbr": "store",
        "family": "product group",
        "dayofweek": "weekday",
        "onpromotion": "promotion"
    }, inplace=True)
    features = [
        "store",
        "product group",
        "weekday",
        "promotion",
    ]

    df_train_full["text"] = ""
    for col in features:
        df_train_full["text"] += col + ": " + df_train_full[col].astype(str) + "\n"

    df_train = df_train_full[df_train_full["date"] <= "2017-07-30"].reset_index()
    df_val = df_train_full[df_train_full["date"] >= "2017-07-31"].reset_index()

    return df_train, df_val


def get_data_house_prices():
    # use data from Kaggle competition https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
    df_train_full = pd.read_csv("train_house_prices.csv")

    df_train_full["target"] = np.log(1 + df_train_full["SalePrice"])

    features = [
        "OverallQual",
        "GarageCars",
        "ExterQual",
        "Neighborhood",
        "GrLivArea",
        "GarageArea",
        "BsmtQual",
        "YearBuilt",
        "KitchenQual",
        "TotalBsmtSF"
    ]

    df_train_full["text"] = ""
    for col in features:
        df_train_full["text"] += col + ": " + df_train_full[col].astype(str) + "\n"

    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)

    return df_train, df_val


def main(args):
    np.random.seed(666)
    torch.manual_seed(42)

    df_train_store_sales, df_val_store_sales = get_data_store_sales()
    df_train_house_prices, df_val_house_prices = get_data_house_prices()

    enc = get_encoder()
    input_ids_train_store_sales = encode_text(df_train_store_sales, enc)
    input_ids_val_store_sales = encode_text(df_val_store_sales, enc)
    input_ids_train_house_prices = encode_text(df_train_house_prices, enc)
    input_ids_val_house_prices = encode_text(df_val_house_prices, enc)

    input_ids_train = input_ids_train_store_sales + input_ids_train_house_prices
    input_ids_val = input_ids_val_store_sales + input_ids_val_house_prices

    max_length = 0
    for ids in input_ids_train:
        len_ids = len(ids)
        if len_ids > max_length:
            max_length = len_ids

    input_ids_train = padding(input_ids_train, max_length)
    input_ids_val = padding(input_ids_val, max_length)

    targets_train = df_train_store_sales["target"].tolist() + df_train_house_prices["target"].tolist()
    targets_val = df_val_store_sales["target"].tolist() + df_val_house_prices["target"].tolist()

    train_dataset = TensorDataset(
        torch.tensor(input_ids_train), 
        torch.tensor(targets_train, dtype=torch.float32)
        )
    val_dataset = TensorDataset(
        torch.tensor(input_ids_val), 
        torch.tensor(targets_val, dtype=torch.float32)
        )

    # create a GPT instance
    if args and args[0] == "--pretrained":
        model = GPT.from_pretrained('gpt2', 1)
    else:
        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        # model_config.model_type = 'gpt2'
        model_config.vocab_size = 50257 # openai's model vocabulary
        model_config.block_size = max_length # 1024 is openai's model block_size
        model_config.n_output_nodes = 1
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
    df_train_store_sales = df_train_store_sales[["target", "text", "date"]]
    df_train_store_sales["dataset"] = "store sales"
    df_train_house_prices = df_train_house_prices[["target", "text"]]
    df_train_house_prices["dataset"] = "house prices"
    df_train_house_prices["date"] = ""
    df_train = pd.concat([df_train_store_sales, df_train_house_prices])
    df_train = predict(model, DataLoader(train_dataset, batch_size=32), df_train)
    evaluation(df_train)
    plot_timeseries(df_train[df_train["dataset"] == "store sales"], "train", True)

    df_val_store_sales = df_val_store_sales[["target", "text", "date"]]
    df_val_store_sales["dataset"] = "store sales"
    df_val_house_prices = df_val_house_prices[["target", "text"]]
    df_val_house_prices["dataset"] = "house prices"
    df_val_house_prices["date"] = ""
    df_val = pd.concat([df_val_store_sales, df_val_house_prices])
    df_val = predict(model, DataLoader(val_dataset, batch_size=32), df_val)
    evaluation(df_val)
    plot_timeseries(df_val[df_val["dataset"] == "store sales"], "val", True)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
