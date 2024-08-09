import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
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
        ts = df.groupby(["Date"])[["sales", "yhat"]].sum().reset_index()
    else:
        ts = df.groupby(["Date"])["sales"].sum().reset_index()
    plt.figure()
    ts.index = ts["Date"]
    ts["sales"].plot(style="r", label="sales")
    if include_preds:
        ts["yhat"].plot(style="b-.", label="predictions")
    plt.legend(fontsize=15)
    plt.ylabel("sum", fontsize=15)
    plt.tight_layout()
    plt.savefig("ts_{}.png".format(suffix))
    plt.clf()


def evaluation(y, yhat):
    print("MSE: ", mean_squared_error(y, yhat))
    print("mean(y): ", np.mean(y))


def predict(model, dataloader, df):
    model.eval()

    yhat = []
    for input_ids, _ in dataloader:
        with torch.no_grad():
            output = (
                model.generate(input_ids.to(device)).cpu().detach().numpy().tolist()
            )
            yhat += np.array(output).squeeze().tolist()

    df["yhat"] = yhat
    df["yhat"] = np.clip(df["yhat"], 0, None)

    return df


def data_preparation(df):
    df.rename(
        columns={
            "P_ID": "product id",
            "PG_ID_3": "product group 3 id",
            "PG_ID_2": "product group 2 id",
            "PG_ID_1": "product group 1 id",
            "NORMAL_PRICE": "normal price",
            "L_ID": "location id",
            "SALES_AREA": "sales area",
            "PROMOTION_TYPE": "type of promotion",
            "SALES_PRICE": "sales price",
            "SALES": "sales",
        },
        inplace=True,
    )

    df["DATE"] = pd.to_datetime(df["DATE"])

    df["month"] = df["DATE"].dt.month_name(locale="English")
    df["year"] = df["DATE"].dt.year
    df["day_of_week"] = df["DATE"].dt.day_name()
    df.drop(["DATE"], axis=1, inplace=True)

    categorical_features = [
        "product id",
        "product group 3 id",
        "product group 2 id",
        "product group 1 id",
        "location id",
        "type of promotion",
        "month",
        "year",
        "day_of_week",
    ]

    numerical_features = ["normal price", "sales area", "sales price"]

    features = categorical_features + numerical_features

    df= df.iloc[:1000]

    return (df, features, categorical_features, numerical_features)


def main(args):
    np.random.seed(666)
    torch.manual_seed(42)

    df_train_sim_low = pd.read_parquet("train_low.parquet.gzip")
    df_train_sim_high = pd.read_parquet("train_high.parquet.gzip")

    (
        df_train_sim_low,
        features_sim_low,
        categorical_features_sim_low,
        numerical_features_sim_low,
    ) = data_preparation(df_train_sim_low)

    (
        df_train_sim_high,
        features_sim_high,
        categorical_features_sim_high,
        numerical_features_sim_high,
    ) = data_preparation(df_train_sim_high)

    features = features_sim_low + features_sim_high

    features_embeds_train_sim_low = get_column_embeddings(
        df_train_sim_low,
        "low sales",
        categorical_features_sim_low,
        numerical_features_sim_low,
        number_of_cols=12,
    )
    features_embeds_train_sim_high = get_column_embeddings(
        df_train_sim_high,
        "high sales",
        categorical_features_sim_high,
        numerical_features_sim_high,
        number_of_cols=12,
    )

    if args and args[0] == "--mode":
        mode = args[1]
    else:
        mode = "train_together"

    if mode == "train_together":
        features_embeds_train = torch.cat(
            (
                features_embeds_train_sim_low,
                features_embeds_train_sim_high,
            ),
            dim=0,
        )

        max_length = len(features) + 2

        targets_train = (
            df_train_sim_low["sales"].tolist() + df_train_sim_high["sales"].tolist()
        )

        train_dataset = TensorDataset(
            features_embeds_train, torch.tensor(targets_train, dtype=torch.float32)
        )

    elif mode == "train_low":
        features_embeds_train = features_embeds_train_sim_low

        max_length = len(features) + 1

        targets_train = df_train_sim_low["sales"].tolist()

    elif mode == "train_high":
        features_embeds_train = features_embeds_train_sim_high

        max_length = len(features) + 1

        targets_train = df_train_sim_high["sales"].tolist()

    else:
        raise Exception("invalid mode")

    train_dataset = TensorDataset(
        features_embeds_train, torch.tensor(targets_train, dtype=torch.float32)
    )

    # tabGPT model
    model_config = tabGPT.get_default_config()
    model_config.model_type = "gpt-nano"
    # model_config.model_type = 'gpt2'
    model_config.vocab_size = 50257  # openai's model vocabulary
    model_config.block_size = max_length  # 1024 is openai's model block_size
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
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

    trainer.set_callback("on_batch_end", batch_end_callback)

    trainer.run()

    # inference
    df_test_sim_low = pd.read_parquet("test_low.parquet.gzip")
    df_test_sim_high = pd.read_parquet("test_high.parquet.gzip")
    df_test_results_sim_low = pd.read_parquet("test_results_low.parquet.gzip")
    df_test_results_sim_high = pd.read_parquet("test_results_high.parquet.gzip")

    (df_test_sim_low, _, _, _,) = data_preparation(df_test_sim_low)
    (df_test_sim_high, _, _, _) = data_preparation(df_test_sim_high)

    features_embeds_test_sim_low = get_column_embeddings(
        df_test_sim_low,
        "low sales",
        categorical_features_sim_low,
        numerical_features_sim_low,
        number_of_cols=12,
    )
    features_embeds_test_sim_high = get_column_embeddings(
        df_test_sim_high,
        "high sales",
        categorical_features_sim_high,
        numerical_features_sim_high,
        number_of_cols=12,
    )

    test_dataset_sim_low = TensorDataset(
        features_embeds_test_sim_low,
        torch.tensor(df_test_results_sim_low["SALES"].iloc[:1000].tolist(), dtype=torch.float32),
    )

    test_dataset_sim_high = TensorDataset(
        features_embeds_test_sim_high,
        torch.tensor(df_test_results_sim_high["SALES"].iloc[:1000].tolist(), dtype=torch.float32),
    )

    df_test_sim_low = predict(
        model, DataLoader(test_dataset_sim_low, batch_size=32), df_test_sim_low
    )
    evaluation(df_test_results_sim_low["SALES"].iloc[:1000], df_test_sim_low["yhat"])
    plot_timeseries(df_test_sim_low, "test", True)

    df_test_sim_high = predict(
        model, DataLoader(test_dataset_sim_high, batch_size=32), df_test_sim_high
    )
    evaluation(df_test_results_sim_high["SALES"].iloc[:1000], df_test_sim_high["yhat"])
    plot_timeseries(df_test_sim_high, "test", True)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
