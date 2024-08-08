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
        ts = df.groupby(["date"])[["target", "yhat"]].sum().reset_index()
    else:
        ts = df.groupby(["date"])["target"].sum().reset_index()
    plt.figure()
    ts.index = ts["date"]
    ts["target"].plot(style="r", label="target")
    if include_preds:
        ts["yhat"].plot(style="b-.", label="predictions")
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
    print("RMSLE: ", root_mean_squared_log_error(y, yhat))
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
    df = backtransform(df)
    return df


def get_data_simulated_low_sales():
    df_train = pd.read_parquet("C:/Users/70Q1985/Downloads/train_low.parquet.gzip")
    df_train.rename(
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

    df_train["DATE"] = pd.to_datetime(df_train["DATE"])

    df_train["month"] = df_train["DATE"].dt.month_name(locale="English")
    df_train["year"] = df_train["DATE"].dt.year
    df_train["day_of_week"] = df_train["DATE"].dt.day_name()
    df_train.drop(["DATE"], axis=1, inplace=True)

    df_train["target"] = np.log(1 + df_train["sales"])

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

    df_train= df_train.iloc[:1000]

    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=666)
    return (df_train, df_val, features, categorical_features, numerical_features)


def get_data_simulated_high_sales():
    df_train = pd.read_parquet("C:/Users/70Q1985/Downloads/train_high.parquet.gzip")
    df_train.rename(
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

    df_train["DATE"] = pd.to_datetime(df_train["DATE"])

    df_train["month"] = df_train["DATE"].dt.month_name(locale="English")
    df_train["year"] = df_train["DATE"].dt.year
    df_train["day_of_week"] = df_train["DATE"].dt.day_name()
    df_train.drop(["DATE"], axis=1, inplace=True)

    df_train["target"] = np.log(1 + df_train["sales"])

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

    df_train= df_train.iloc[:1000]

    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=666)
    return (df_train, df_val, features, categorical_features, numerical_features)


def main(args):
    np.random.seed(666)
    torch.manual_seed(42)

    (
        df_train_sim_low,
        df_val_sim_low,
        features_sim_low,
        categorical_features_sim_low,
        numerical_features_sim_low,
    ) = get_data_simulated_low_sales()

    (
        df_train_sim_high,
        df_val_sim_high,
        features_sim_high,
        categorical_features_sim_high,
        numerical_features_sim_high,
    ) = get_data_simulated_high_sales()

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

    features_embeds_train = torch.cat(
        (
            features_embeds_train_sim_low,
            features_embeds_train_sim_high,
        ),
        dim=0,
    )

    max_length = len(features)

    targets_train = (
        df_train_sim_low["target"].tolist() + df_train_sim_high["target"].tolist()
    )

    train_dataset = TensorDataset(
        features_embeds_train, torch.tensor(targets_train, dtype=torch.float32)
    )

    # tabGPT model
    if args and args[0] == "--pretrained":
        model = tabGPT.from_pretrained("gpt2", 1)
    else:
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
    features_embeds_val_sim_low = get_column_embeddings(
        df_val_sim_low,
        "low sales",
        categorical_features_sim_low,
        numerical_features_sim_low,
        number_of_cols=12,
    )
    features_embeds_val_sim_high = get_column_embeddings(
        df_val_sim_high,
        "high sales",
        categorical_features_sim_high,
        numerical_features_sim_high,
        number_of_cols=12,
    )

    val_dataset_sim_low = TensorDataset(
        features_embeds_val_sim_low,
        torch.tensor(df_val_sim_low["target"].tolist(), dtype=torch.float32),
    )

    val_dataset_sim_high = TensorDataset(
        features_embeds_val_sim_high,
        torch.tensor(df_val_sim_high["target"].tolist(), dtype=torch.float32),
    )

    df_val_sim_low = predict(
        model, DataLoader(val_dataset_sim_low, batch_size=32), df_val_sim_low
    )
    evaluation(df_val_sim_low["target"], df_val_sim_low["yhat"])
    plot_timeseries(df_val_sim_low, "val", True)

    df_val_sim_high = predict(
        model, DataLoader(val_dataset_sim_high, batch_size=32), df_val_sim_high
    )
    evaluation(df_val_sim_high["target"], df_val_sim_high["yhat"])
    plot_timeseries(df_val_sim_high, "val", True)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
