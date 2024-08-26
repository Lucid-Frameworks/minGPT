import torch

from transformers import GPT2Model, AutoTokenizer


if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def get_column_embeddings(df, target_name, categorical_features, numerical_features,
                          number_of_cols=10,
                          null_treatment="zero-embedding", fillna_categorical="missing value", fillna_numerical=0):
    features = categorical_features + numerical_features
    number_of_features = len(features) + 1
    number_of_cols += 1
    assert number_of_features <= number_of_cols, "total number of features must not be larger than set number_of_cols"

    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    inputs_embeds = torch.empty(len(df), 1, 768)

    df[categorical_features] = df[categorical_features].fillna(fillna_categorical)
    df[numerical_features] = df[numerical_features].fillna(fillna_numerical)

    for col in features:
        input_ids = tokenizer(col, return_tensors="pt")
        with torch.no_grad():
            colname_embed = model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu()
        colname_embed = colname_embed.repeat(len(df), 1)

        if col in categorical_features:
            cat_embed_dict = {}
            for category in df[col].unique().tolist():
                input_ids = tokenizer(str(category), return_tensors="pt")
                with torch.no_grad():
                    cat_embed_dict[category] = model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu()

            cat_embeds = torch.empty(1, 768)
            for i in range(len(df)):            
                cat_embeds = torch.cat((cat_embeds, cat_embed_dict[df[col].iloc[i]]))

            col_embeds = colname_embed + cat_embeds[1:]
        else:
            col_values = torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1)

            if null_treatment == "shift":
                col_embeds = colname_embed * torch.where(col_values >= 0, col_values + 1, col_values - 1)
            elif null_treatment == "zero-embedding":
                col_values = torch.where(col_values == 0, 1, col_values)
                col_embeds = colname_embed * col_values

                input_ids = tokenizer(str(0), return_tensors="pt")
                with torch.no_grad():
                    cat0_embed = model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu()
                cat_embeds = torch.zeros(len(df), 768)
                for i in range(len(df)):
                    if df[col].iloc[i] == 0:
                        cat_embeds[i, :] = cat0_embed
                col_embeds = col_embeds + cat_embeds
            elif "simple":
                col_embeds = colname_embed * col_values
            else:
                raise ValueError

        inputs_embeds = torch.cat((inputs_embeds, col_embeds.unsqueeze(1)), dim=1)
    
    inputs_embeds = inputs_embeds[:, 1:, :]

    input_ids = tokenizer(target_name, return_tensors="pt")
    with torch.no_grad():
        target_embed = model(**input_ids.to(device)).last_hidden_state.mean(dim=1).cpu()
    target_embed = target_embed.repeat(len(df), 1, 1) # nrows, 1, emb_dim

    features_embeds_wo_pos = torch.cat((target_embed, inputs_embeds), dim=1)
    rows, features, emb_dim = features_embeds_wo_pos.shape
    features_embeds = torch.zeros(rows, features, emb_dim)
    features_embeds[:, 1:, :] = 1.
    features_embeds += features_embeds_wo_pos

    if number_of_features < number_of_cols:
        padding_features_embeds = torch.empty(len(df), number_of_cols - number_of_features, 768)
        features_embeds = torch.cat((features_embeds, padding_features_embeds), dim=1)

    return features_embeds
