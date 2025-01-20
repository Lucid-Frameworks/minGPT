# Mechanistic Interpretability with GPT-2 Model

## by Gabriel Masella and Laihi Bahar Eddine

### Assignmemt for NLP Master's Course of AI of University of Alicante

### Problem Statement

The aim of this assignment is to explore mechanistic interpretability in transformer models by implementing activation patching on a GPT2 model. The task involves running GPT2 with two inputs, a clean text and a corrupted one, differing by a single token and analyzing the change in model output probabilities. By intervening in specific model activations and comparing the probabilities, it possible to gain insights into how individual activations influence the model's predictions.

### Chosen Approach

### Implementation

For this project a PyTorch re-implementation of [GPT](https://github.com/openai/gpt-2), both training and inference, is used called [**minGPT**](https://github.com/karpathy/minGPT) that tries to be small, clean, interpretable and educational, as most of the currently available GPT model implementations can a bit sprawling. GPT is not a complicated model and this implementation is appropriately about 300 lines of code (see [mingpt/model.py](mingpt/model.py)). All that's going on is that a sequence of indices feeds into a [Transformer](https://arxiv.org/abs/1706.03762), and a probability distribution over the next index in the sequence comes out.

The minGPT library is three files: [mingpt/model.py](mingpt/model.py) contains the actual Transformer model definition, [mingpt/bpe.py](mingpt/bpe.py) contains a mildly refactored Byte Pair Encoder that translates between text and sequences of integers exactly like OpenAI did in GPT, [mingpt/trainer.py](mingpt/trainer.py) is (GPT-independent) PyTorch boilerplate code that trains the model.

The implementation involved significant modifications to the GPT model architecture to enable interpretability analysis through activation patching.  
The main changes focused on expanding the forward pass of the model to support two key functionalities: activation storage and targeted patching. A new `layer_activations` dictionary attribute was introduced to store intermediate activations at each transformer layer, with each activation tensor properly detached and cloned to prevent unwanted gradient flow and ensure data persistence. A `store_activations` parameter has been added to the forward method which, when enabled, captures and stores these layer-wise activations. In addition, a `patch_params` parameter was implemented to allow precise intervention at specific layers and positions, allowing the substitution of activations from one forward pass to another. This modification facilitates comparative analysis between different model runs. The implementation also includes the storage of the last token's logits through the `last_token_logits` attribute, which is crucial for analysing the model's final predictions.  
These architectural changes preserve the original functionality of the model while adding the necessary infrastructure for detailed mechanistic interpretability studies.

### Results

### Conclusion
