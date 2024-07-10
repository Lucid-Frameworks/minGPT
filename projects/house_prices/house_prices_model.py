import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.bpe import get_encoder
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


def main(pretrained, enrich, validation):
    np.random.seed(666)
    torch.manual_seed(42)

    if enrich:
        df_train_full, df_test = construct_text()
    else:
        # use data from Kaggle competition https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
        df_train_full = pd.read_csv("train.csv")
        df_test = pd.read_csv("test.csv")

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
        df_test["text"] = ""
        for col in features:
            df_train_full["text"] += col + ": " + df_train_full[col].astype(str) + "\n"
            df_test["text"] += col + ": " + df_test[col].astype(str) + "\n"

    df_train_full["SalePrice_transformed"] = np.log(1 + df_train_full["SalePrice"])

    df_train_full = df_train_full[['Id', 'text', 'SalePrice', 'SalePrice_transformed']]
    df_test = df_test[['Id', 'text']]

    if validation:
        df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)
    else:
        df_train = df_train_full

    enc = get_encoder()
    input_ids_train = encode_text(df_train, enc)
    input_ids_test = encode_text(df_test, enc)

    max_length = 0
    for ids in input_ids_train:
        len_ids = len(ids)
        if len_ids > max_length:
            max_length = len_ids
    for ids in input_ids_test:
        len_ids = len(ids)
        if len_ids > max_length:
            max_length = len_ids

    input_ids_train = padding(input_ids_train, max_length)

    train_dataset = TensorDataset(
        torch.tensor(input_ids_train), 
        torch.tensor(df_train["SalePrice_transformed"].tolist(), dtype=torch.float32)
        )

    # create a GPT instance
    if pretrained:
        model = GPT.from_pretrained('gpt2')
    else:
        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        # model_config.model_type = 'gpt2'
        model_config.vocab_size = 50257 # openai's model vocabulary
        model_config.block_size = max_length # 1024 is openai's model block_size
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
    evaluation(df_train["SalePrice"], df_train["yhat"])

    if validation:
        input_ids_val = encode_text(df_val, enc)
        input_ids_val = padding(input_ids_val, max_length)
        val_dataset = TensorDataset(
            torch.tensor(input_ids_val), 
            torch.tensor(df_val["SalePrice_transformed"].tolist(), dtype=torch.float32)
            )
        df_val = predict(model, DataLoader(val_dataset, batch_size=32), df_val)
        evaluation(df_val["SalePrice"], df_val["yhat"])
    else:
        input_ids_test = padding(input_ids_test, max_length)
        test_dataset = TensorDataset(
            torch.tensor(input_ids_test),
            torch.tensor(df_test["Id"].tolist(), dtype=torch.float32)
            )
        df_test = predict(model, DataLoader(test_dataset, batch_size=32), df_test)
        df_test.rename(columns={"yhat": "SalePrice"}, inplace=True)
        df_test[["Id", "SalePrice"]].to_csv("submission.csv", index=False)

    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--enrich", action="store_true")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    main(args.pretrained, args.enrich, args.validation)
