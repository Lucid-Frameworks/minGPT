#CLEAN_INPUT = " Cristiano Ronaldo and Lionel Messi have both left an indelible mark on football through their incredible careers. Ronaldo has played for top clubs like Manchester United, Real Madrid, and Juventus, and currently plays for Al-Nassr. He has won numerous titles, including five Champions League titles and five Ballon d'Or awards. Messi, on the other hand, has spent the majority of his career at FC Barcelona before moving to Paris Saint-Germain. He has also achieved great success, with four Champions League titles and an impressive seven Ballon d'Or awards. As of now, Ronaldo has a total goal contribution (goals + assists) of 1027, while Messi's goal contribution stands at 1092, showcasing the extraordinary talent and dedication of both players. "
#END = "The best player in the world is undoubtedly"
#CORRUPTED_INPUT = " Cristiano Ronaldo and Lionel Messi have both left an indelible mark on football through their incredible careers. Ronaldo has played for top clubs like Manchester United, Real Madrid, and Juventus, and currently plays for Al-Nassr. He has won numerous titles, including five Champions League titles and five Ballon d'Or awards. Messi, on the other hand, has spent the majority of his career at FC Barcelona before moving to Paris Saint-Germain. He has also achieved great success, with four Champions League titles and an impressive seven Ballon d'Or awards. As of now, Ronaldo has a total goal contribution (goals + assists) of 1027, while Messi's goal contribution stands at 1011, showcasing the extraordinary talent and dedication of both players. "

CLEAN_INPUT = " Michelle Jones was a top-notch student."
END = " Michelle"
CORRUPTED_INPUT = " Michelle Smith was a top-notch student."

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch as th
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
import torch.nn.functional as F

def get_top_predictions(logits, tokenizer, k=20):
    """Get top k predictions from logits"""
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = th.topk(probs, k)
    
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        # Convert single index to tensor before decoding
        idx_tensor = th.tensor([idx.item()])
        token = tokenizer.decode(idx_tensor)
        predictions.append((token, prob.item()))
    return predictions

device = "mps" if th.backends.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu"
print(f"Running on device: {device}")
model_type = "gpt2-medium"  # Changed to smallest model for testing

# Initialize model
model = GPT.from_pretrained(model_type)
model.to(device)
model.eval()

# Tokenize inputs
tokenizer = BPETokenizer()
clean = tokenizer(CLEAN_INPUT + END).to(device)
corrupted = tokenizer(CORRUPTED_INPUT + END).to(device)  # Also move to device

# Get predictions for clean input
logits_clean, _ = model(clean, store_activations=True)
clean_activations = model.layer_activations.copy()
print("\nTop predictions for clean input:")
print(get_top_predictions(model.last_token_logits, tokenizer))

# Get reference probability for specific tokens
# messi_idx = tokenizer(" Messi")[0]  
# ronaldo_idx = tokenizer(" Ronaldo")[0]
smith_idx = tokenizer(" Jones")[0]
reference_logits = model.last_token_logits[0]
smith_prob = reference_logits[smith_idx].item()
# messi_prob = reference_logits[messi_idx].item()
# ronaldo_prob = reference_logits[ronaldo_idx].item()

# Setup patching loop
n_layers = len(model.transformer.h)
print(f"Number of layers: {n_layers}")
seq_length = corrupted.size(1)
print(f"Sequence length: {seq_length}")
diff_matrix = np.zeros((n_layers, seq_length))

# Get tokens for x-axis labels
tokens = [tokenizer.decode(th.tensor([corrupted[0, i].item()])) for i in range(seq_length)]

# Iterate through layers and positions
for layer in range(n_layers):
    for pos in range(seq_length):
        # Forward pass with patching
        logits_patched, _ = model(corrupted.to(device), 
                                patch_params=(layer, pos, clean_activations[f'layer_{layer}'][:, pos, :]))
        
        # Convert logits to probabilities
        clean_probs = F.softmax(reference_logits, dim=-1)
        patched_probs = F.softmax(model.last_token_logits[0], dim=-1)
        
        # Compute probability difference
        diff_matrix[layer, pos] = patched_probs[smith_idx].item() - clean_probs[smith_idx].item()

# Get predictions for corrupted input
logits_corrupted, _ = model(corrupted.to(device), patch_params=(n_layers,seq_length, clean_activations[f'layer_{n_layers-1}'][:, seq_length-1, :]))
print("\nTop predictions for corrupted input:")
print(get_top_predictions(model.last_token_logits, tokenizer))

# Visualize results with token labels
plt.figure(figsize=(30, 16))
sns.heatmap(diff_matrix, cmap='crest', annot=True, xticklabels=tokens)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Token')
plt.ylabel('Layer')
plt.title("Patching Heatmap of 'Jones' Token in the Corrupted Input")
plt.tight_layout()
plt.savefig(f'patching_heatmap_{model_type}.png')
plt.close()