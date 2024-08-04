import torch

from transformers import GPT2Model, AutoTokenizer


if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def get_column_embeddings(df, categorical_features, numerical_features, number_of_cols=10):
    number_of_features = len(categorical_features + numerical_features)
    assert number_of_features <= number_of_cols, "total number of features must not be larger than set number_of_cols"

    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    cat_features_embeds = torch.empty(len(df), 1, 768)
    for col in categorical_features:
    # for col in categorical_features + numerical_features:
        input_ids = tokenizer(col, return_tensors="pt", truncation=True)
        colname_embed = model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu().detach()

        df[col] = df[col].astype(str)
        col_embeds = torch.empty(1, 768)
        col_values = df[col].tolist()
        for i in range(len(col_values)):
            input_ids = tokenizer(col_values[i], return_tensors="pt", truncation=True)
            col_embeds = torch.cat((col_embeds, model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu().detach()))
        col_embeds = col_embeds[1:]

        col_embeds = colname_embed.repeat(len(df), 1) + col_embeds
        col_embeds = col_embeds.unsqueeze(1)

        cat_features_embeds = torch.cat((cat_features_embeds, col_embeds), dim=1)
    cat_features_embeds = cat_features_embeds[:, 1:, :]

    # return cat_features_embeds
    num_features_embeds = torch.empty(len(df), 1, 768)
    for col in numerical_features:
        input_ids = tokenizer(col, return_tensors="pt", truncation=True)
        colname_embed = model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu().detach()

        col_embeds = colname_embed.repeat(len(df), 1)
        col_embeds = col_embeds * torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1)
        col_embeds = col_embeds.unsqueeze(1)

        num_features_embeds = torch.cat((num_features_embeds, col_embeds), dim=1)
    num_features_embeds = num_features_embeds[:, 1:, :]

    features_embeds = torch.cat((cat_features_embeds, num_features_embeds), dim=1)

    if number_of_features < number_of_cols:
        padding_features_embeds = torch.empty(len(df), number_of_cols - number_of_features, 768)
        features_embeds = torch.cat((features_embeds, padding_features_embeds), dim=1)

    return features_embeds
