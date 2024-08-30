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
        ts = df.groupby(['DATE'])[['SALES', 'yhat']].sum().reset_index()
    else:
        ts = df.groupby(['DATE'])['SALES'].sum().reset_index()
    plt.figure()
    ts.index = ts['DATE']
    ts['SALES'].plot(style='r', label="sales")
    if include_preds:
        ts['yhat'].plot(style='b-.', label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("ts_{}.png".format(suffix))
    plt.clf()


def evaluation(y, yhat):
    print("RMSLE: ", root_mean_squared_log_error(y, yhat))
    print("mean(y): ", np.mean(y))


def predict(model, dataloader, df):
    model.eval()

    yhat = []
    for input_ids, _ in dataloader:
        with torch.no_grad():
            yhat += model.generate(input_ids.to(device)).cpu().detach().numpy().tolist()

    df["yhat"] = yhat
    df["yhat"] = np.clip(df["yhat"], 0, None)
    df["yhat"] = np.exp(df["yhat"]) - 1
    return df


def ewma_prediction(df, group_cols, col, alpha, horizon):
    df.sort_values(["DATE"], inplace=True)
    df_grouped = df.groupby(group_cols, group_keys=False)
    df["past sales"] = df_grouped[col].apply(lambda x: x.shift(horizon).ewm(alpha=alpha, ignore_na=True).mean())
    return df


def ewma_merge(df_test, df_train, ewma_col, group_cols):
    def get_latest_ewmas(df):
        return df.loc[df["DATE"] == df["DATE"].max(), ewma_col]

    df_train_latest_ewma = df_train[["DATE", ewma_col] + group_cols].groupby(group_cols).apply(get_latest_ewmas).reset_index()

    df_test = df_test.merge(df_train_latest_ewma[[ewma_col] + group_cols], on=group_cols, how="left")

    return df_test


def data_preparation(df):
    df.rename(
        columns={
            "P_ID": "product id",
            "PG_ID_3": "product group id",
            "NORMAL_PRICE": "normal price",
            "L_ID": "location id",
            "SALES_AREA": "sales area",
            "PROMOTION_TYPE": "type of promotion",
            "SALES_PRICE": "sales price",
        },
        inplace=True,
    )

    df["DATE"] = pd.to_datetime(df["DATE"])
    df["weekday"] = df['DATE'].dt.day_name()
    df["day in month"] = df['DATE'].dt.day
    df["day in year"] = df['DATE'].dt.dayofyear

    return df


def main(args):
    np.random.seed(666)
    torch.manual_seed(42)

    df_train = pd.read_parquet("train.parquet.gzip")

    df_train = data_preparation(df_train)

    df_train["sales_transformed"] = np.log(1 + df_train["SALES"])

    ewma_groups = ["location id", "product id", "weekday"]
    df_train = ewma_prediction(df_train, ewma_groups, "sales_transformed", 0.15, 1)

    categorical_features = [
        "product id",
        "product group id",
        "location id",
        "type of promotion",
        "weekday",
    ]
    numerical_features = [
        "normal price",
        "sales area", 
        "sales price",
        "day in month",
        "day in year",
        "past sales",
    ]

    num_max = df_train[numerical_features].abs().max()

    features = categorical_features + numerical_features

    df_train[numerical_features] = df_train[numerical_features] / num_max
    features_embeds_train = get_column_embeddings(
        df_train,
        "sales",
        categorical_features,
        numerical_features,
        number_of_cols=len(features),
    )

    max_length = len(features) + 1

    targets_train = df_train["sales_transformed"].tolist()

    train_dataset = TensorDataset(
        features_embeds_train, torch.tensor(targets_train, dtype=torch.float32)
    )

    # tabGPT model
    model_config = tabGPT.get_default_config()
    model_config.model_type = "gpt-micro"
    model_config.vocab_size = 50257  # openai's model vocabulary
    model_config.block_size = max_length  # 1024 is openai's model block_size
    model_config.n_output_nodes = 1
    model = tabGPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.max_iters = 100000
    train_config.epochs = 100
    train_config.num_workers = 0
    train_config.batch_size = 64
    train_config.observe_train_loss = True
    trainer = Trainer(train_config, model, train_dataset)

    if train_config.observe_train_loss:
        def epoch_end_callback(trainer):
            print(f"epoch {trainer.epoch}: train loss {np.sqrt(trainer.aggregated_loss.detach().cpu())}")
        trainer.set_callback('on_epoch_end', epoch_end_callback)
    else:
        def batch_end_callback(trainer):
            if trainer.iter_num % 100 == 0:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

    # inference
    df_test = pd.read_parquet("test.parquet.gzip")
    df_test_results = pd.read_parquet("test_results.parquet.gzip")

    df_test = data_preparation(df_test)

    df_test = ewma_merge(df_test, df_train, "past sales", ewma_groups)

    df_test[numerical_features] = df_test[numerical_features] / num_max
    features_embeds_test = get_column_embeddings(
        df_test,
        "low sales",
        categorical_features,
        numerical_features,
        number_of_cols=len(features),
    )

    test_dataset = TensorDataset(
        features_embeds_test,
        torch.tensor(df_test_results["SALES"].tolist(), dtype=torch.float32),
    )

    df_test = predict(
        model, DataLoader(test_dataset, batch_size=32), df_test
    )
    evaluation(df_test_results["SALES"], df_test["yhat"])
    df_test = df_test.merge(df_test_results, left_on=['product id', 'location id', 'DATE'], right_on=['P_ID', 'L_ID', 'DATE'])
    plot_timeseries(df_test, "val", True)
    for pg in df_test["product group id"].unique():
        plot_timeseries(df_test[df_test["product group id"] == pg], str(pg) + "_val", True)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
