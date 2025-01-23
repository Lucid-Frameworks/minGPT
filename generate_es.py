import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch as th
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
import torch.nn.functional as F
from mingpt.utils import set_seed

# Define the clean and corrupted sentences in Spanish
FACT_CLEAN = "El gato se sentó en la alfombra mientras el perro ladraba ruidosamente, y el sol brillaba en el cielo."
FACT_CORRUPTED = FACT_CLEAN.replace("ruidosamente", "silenciosamente")
END = ""
SPECIFIC_TOKENS = ["gato", "perro", "ladraba", "ruidosamente", "silenciosamente"]

def get_specific_token_probs(logits, tokenizer, tokens):
    """Get probabilities of specific tokens from logits, handling multi-token cases."""
    probs = F.softmax(logits, dim=-1)  # Apply softmax to logits
    token_probs = {}

    for token in tokens:
        token_ids = tokenizer(token)[0]  # Get token IDs for the word
        token_ids = token_ids.tolist() if isinstance(token_ids, th.Tensor) else token_ids

        # Sum probabilities for all subword tokens
        token_prob = sum(probs[0, token_id].item() for token_id in token_ids)
        token_probs[token] = token_prob

    return token_probs

def generate_heatmap(model_type, diff_matrix, tokens, specific_token):
    plt.figure(figsize=(30, 16))
    sns.heatmap(diff_matrix,
                cmap='crest',
                annot=True,
                xticklabels=tokens)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Token')
    plt.ylabel('Layer')
    plt.title(f"Patching Heatmap of '{specific_token}' Token in the Corrupted Input")
    plt.tight_layout()
    
    # Save to a specific directory
    output_dir = "./heatmaps/"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    plt.savefig(f'{output_dir}patching_heatmap_{model_type}_{specific_token}.png')
    plt.close()

def tokenize_and_print(tokenizer, text, device):
    tokens = tokenizer(text).to(device)
    tokens_str = [tokenizer.decode(th.tensor([token])) for token in tokens[0]]
    print("Detokenized input as strings: " + '/'.join(tokens_str))
    return tokens

# Pad the shorter sequence to match the longer one
def pad_sequences(clean, corrupted):
    max_length = max(clean.size(1), corrupted.size(1))
    padding = lambda seq, length: th.cat([seq, th.zeros(1, length - seq.size(1), dtype=th.long)], dim=1)
    clean_padded = clean if clean.size(1) == max_length else padding(clean, max_length)
    corrupted_padded = corrupted if corrupted.size(1) == max_length else padding(corrupted, max_length)
    return clean_padded, corrupted_padded

# Initialization
device = "mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu"
print(f"Running on device: {device}")
model_type = "gpt2-medium"
seed = 78
set_seed(seed)

# Initialize model, tokenizer and input
model = GPT.from_pretrained(model_type)
model.to(device)
model.eval()
tokenizer = BPETokenizer()

# Tokenize inputs
clean = tokenize_and_print(tokenizer, FACT_CLEAN + END, device)
corrupted = tokenize_and_print(tokenizer, FACT_CORRUPTED + END, device)

# Pad clean and corrupted inputs to ensure equal sequence length
clean, corrupted = pad_sequences(clean, corrupted)
print(f"Clean sequence length after padding: {clean.size(1)}")
print(f"Corrupted sequence length after padding: {corrupted.size(1)}")

# Get predictions for clean input
logits_clean, _ = model(clean, store_activations=True)
clean_activations = model.layer_activations.copy()
print("\nProbabilities for specific tokens in clean input:")
print(get_specific_token_probs(model.last_token_logits, tokenizer, SPECIFIC_TOKENS))
reference_logits = model.last_token_logits[0]

# Setup patching loop
n_layers = len(model.transformer.h)
print(f"Number of layers: {n_layers}")
seq_length = corrupted.size(1)
print(f"Sequence length: {seq_length}")

# Get tokens
tokens = [tokenizer.decode(th.tensor([corrupted[0, i].item()])) for i in range(seq_length)]

# Iterate through each specific token
for specific_token in SPECIFIC_TOKENS:
    print(f"Processing token: {specific_token}")
    diff_matrix = np.zeros((n_layers, seq_length))
    
    # Get token IDs for clean and corrupted input
    token_ids_clean = tokenizer(specific_token)[0]
    token_ids_clean = token_ids_clean.tolist() if isinstance(token_ids_clean, th.Tensor) else token_ids_clean

    token_ids_corrupted = tokenizer(specific_token.replace("rumorosamente", "silenziosamente"))[0]
    token_ids_corrupted = token_ids_corrupted.tolist() if isinstance(token_ids_corrupted, th.Tensor) else token_ids_corrupted

    # Iterate through layers and positions
    for layer in range(n_layers):
        for pos in range(seq_length):
            # Forward pass with patching
            logits_patched, _ = model(
                corrupted.to(device),
                patch_params=(layer, pos, clean_activations[f'layer_{layer}'][:, pos, :])
            )

            # Convert logits to probabilities
            patched_probs = F.softmax(model.last_token_logits[0], dim=-1)
            
            # Compute probabilities for clean and corrupted
            clean_prob = sum(reference_logits[token_id].item() for token_id in token_ids_clean)
            patched_prob = sum(patched_probs[token_id].item() for token_id in token_ids_corrupted)

            # Compute the probability difference
            diff_matrix[layer, pos] = patched_prob - clean_prob

    # Generate heatmap
    generate_heatmap(model_type, diff_matrix, tokens, specific_token)

# Get predictions for corrupted input
logits_corrupted, _ = model(corrupted.to(device))
print("\nProbabilities for specific tokens in corrupted input:")
print(get_specific_token_probs(logits_corrupted[0], tokenizer, SPECIFIC_TOKENS))
