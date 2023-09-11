import streamlit as st
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
from torch.nn import functional as F


st.title('GoLem Pharm')
st.write('')

#input box
int_put2 =  st.text_input('Sequence of protein:   e.g. MLLETQDALYVALELVIAALSVAGNVLVCAAVG...')

batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
##with open("./seqsmile.txt", 'r', encoding='utf-8') as f:
##    text = f.read()

# here are all the unique characters that occur in this text
##chars = sorted(list(set(text)))
vocab_size = 71
# create a mapping from characters to integers
#stoi = { ch:i for i,ch in enumerate(chars) }
#itos = { i:ch for i,ch in enumerate(chars) }
stoi = {'\n': 0, ' ': 1, '#': 2, '(': 3, ')': 4,
 '+': 5, '-': 6, '.': 7, '/': 8, '1': 9,
 '2': 10, '3': 11, '4': 12, '5': 13, '6': 14,
 '7': 15, '8': 16, '9': 17, '=': 18, '@': 19,
 'A': 20, 'B': 21, 'C': 22, 'D': 23, 'E': 24,
 'F': 25, 'G': 26, 'H': 27, 'I': 28, 'K': 29,
 'L': 30, 'M': 31, 'N': 32, 'O': 33, 'P': 34,
 'Q': 35, 'R': 36, 'S': 37, 'T': 38, 'U': 39,
 'V': 40, 'W': 41, 'X': 42, 'Y': 43, 'Z': 44,
 '[': 45, '\\': 46, ']': 47, 'a': 48, 'b': 49,
 'c': 50, 'd': 51, 'e': 52, 'f': 53, 'g': 54,
 'h': 55, 'i': 56, 'k': 57, 'l': 58, 'm': 59,
 'n': 60, 'o': 61, 'p': 62, 'q': 63, 'r': 64,
 's': 65, 't': 66, 'u': 67, 'v': 68, 'w': 69,
 'y': 70}
itos = {0: '\n', 1: ' ', 2: '#', 3: '(', 4: ')',
 5: '+', 6: '-', 7: '.', 8: '/', 9: '1',
 10: '2', 11: '3', 12: '4', 13: '5', 14: '6',
 15: '7', 16: '8', 17: '9', 18: '=', 19: '@',
 20: 'A', 21: 'B', 22: 'C', 23: 'D', 24: 'E',
 25: 'F', 26: 'G', 27: 'H', 28: 'I', 29: 'K',
 30: 'L', 31: 'M', 32: 'N', 33: 'O', 34: 'P',
 35: 'Q', 36: 'R', 37: 'S', 38: 'T', 39: 'U',
 40: 'V', 41: 'W', 42: 'X', 43: 'Y', 44: 'Z',
 45: '[', 46: '\\', 47: ']', 48: 'a', 49: 'b',
 50: 'c', 51: 'd', 52: 'e', 53: 'f', 54: 'g',
 55: 'h', 56: 'i', 57: 'k', 58: 'l', 59: 'm',
 60: 'n', 61: 'o', 62: 'p', 63: 'q', 64: 'r',
 65: 's', 66: 't', 67: 'u', 68: 'v', 69: 'w',
 70: 'y'}

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
##data = torch.tensor(encode(text), dtype=torch.long)
##n = int(0.9*len(data)) # first 90% will be train, rest val
##train_data = data[:n]
##val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
#load model
model = BigramLanguageModel()
m = model.to(device)
m.load_state_dict(torch.load('./ms.pt'))
m.eval()

if int_put2:
 with st.spinner('Please wait...'):
    data = torch.tensor(encode(str(int_put2)), dtype=torch.long, device=device)
    data2d = data.view(1, -1)
    input_protein = []
    generated_smile = []
    for i in range(0,3):
        g = decode(m.generate(data2d, max_new_tokens=1000)[0].tolist())
        gg = g.split()
        generated_smile.append(gg[1])
        input_protein.append(gg[0])
        i+=1

    df = pd.DataFrame()
    df['generated_smile'] = generated_smile
    df['input_protein'] = input_protein
    df['to_ana'] = df['generated_smile'] + ' ' + df['input_protein']
    #st.dataframe(df['generated_smile'].style.format({'value (pKi)':'{:.2f}'}))
    st.dataframe(df['generated_smile'])
