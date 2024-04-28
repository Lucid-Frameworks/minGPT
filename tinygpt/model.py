"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

from tinygrad import dtypes
from tinygrad.tensor import Tensor
import tinygrad.nn as nn

from tinygpt import tinyutils
from tinygpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

# NOTE: Tinygrad layer initialization doesn't support customizing the mean and std.

class NewGELU:
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __call__(self, x):
        return 0.5 * x * (1.0 + (math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3.0))).tanh())

class CausalSelfAttention:
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.bias = Tensor.ones(config.block_size, config.block_size).tril().view(1, 1, config.block_size, config.block_size)
        self.bias.requires_grad = False
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.softmax(axis=-1)
        att = att.dropout(self.attn_pdrop)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y).dropout(self.resid_pdrop)
        return y

class Block:
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.resid_pdrop = config.resid_pdrop
        self.mlp = dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        )
        m = self.mlp
        self.mlpf = lambda x: m['c_proj'](m['act'](m['c_fc'](x))).dropout(self.resid_pdrop) # MLP forward

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT:
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = config.embd_pdrop, # in Tinygrad dropout is a function not a Layer.
            h = [Block(config) for _ in range(config.n_layer)],
            ln_f = nn.LayerNorm(config.n_embd),
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        # NOTE: Tinygrad doesn't support custom initialization

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in nn.state.get_parameters(self.transformer))
        print("number of parameters: %.2fM" % (n_params/1e6,))

    # @classmethod
    # def from_pretrained(cls, model_type):
    #     """
    #     Initialize a pretrained GPT model by copying over the weights
    #     from a huggingface/transformers checkpoint.
    #     """
    #     assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    #     from transformers import GPT2LMHeadModel

    #     # create a from-scratch initialized minGPT model
    #     config = cls.get_default_config()
    #     config.model_type = model_type
    #     config.vocab_size = 50257 # openai's model vocabulary
    #     config.block_size = 1024  # openai's model block_size
    #     model = GPT(config)
    #     sd = model.state_dict()

    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()

    #     # copy while ensuring all of the parameters are aligned and match in names and shapes
    #     keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
    #     transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
    #     # this means that we have to transpose these weights when we import them
    #     assert len(keys) == len(sd)
    #     for k in keys:
    #         if any(k.endswith(w) for w in transposed):
    #             # special treatment for the Conv1D weights we need to transpose
    #             assert sd_hf[k].shape[::-1] == sd[k].shape
    #             # with torc.no_grad():
    #             Tensor.no_grad = True
    #             sd[k].copy_(sd_hf[k].t())
    #             Tensor.no_grad = False
    #         else:
    #             # vanilla copy over the other parameters
    #             assert sd_hf[k].shape == sd[k].shape
    #             # with torc.no_grad():
    #             Tensor.no_grad = True
    #             sd[k].copy_(sd_hf[k])
    #             Tensor.no_grad = False

    #     return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the optimizer object.
        """

        # NOTE: Tinygrad's AdamW doesn't support decay. hmmm...
        params = list(filter(lambda x: x.requires_grad != False, nn.state.get_parameters(self))) # Do not optimize the requires_grad=False tensors
        optimizer = nn.optim.AdamW(params, lr=train_config.learning_rate, b1=train_config.betas[0], b2=train_config.betas[1])
        return optimizer

    def __call__(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = Tensor.arange(0, t, dtype=dtypes.long).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer['wte'](idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer['wpe'](pos) # position embeddings of shape (1, t, n_embd)
        x = (tok_emb + pos_emb).dropout(self.transformer['drop'])
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = logits.view(-1, logits.size(-1)).sparse_categorical_crossentropy( targets.view(-1), ignore_index=-1)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        Tensor.no_grad = True
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = tinyutils.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = logits.softmax(axis=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = Tensor.multinomial(probs, num_samples=1)
            else:
                _, idx_next = tinyutils.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = Tensor.cat((idx, idx_next), dim=1)

        Tensor.no_grad = False

        return idx
