import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# manual seed
torch.manual_seed(1337)

# Read the data
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# define the vocabulary
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# define the mapping from char to index and vice versa
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = {i: u for i, u in enumerate(vocab)}
encode = lambda x: [char2idx[c] for c in x]
decode = lambda x: "".join([idx2char[c] for c in x])

# define the dataset
encoded_text = encode(text)
data = torch.tensor(encoded_text, dtype=torch.long)
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]


# define the dataloader
def get_batch(split):
    data = train_data if split == "train" else val_data
    # randomly select the starting indices for the examples in the mini-batch
    start_indices = torch.randint(len(data) - block_size, size=(batch_size,))
    # select the contiguous sequences of tokens
    x = torch.stack([data[i : i + block_size] for i in start_indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in start_indices])
    # move the data to the device
    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        total_loss = 0
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            total_loss += loss.item()
        out[split] = total_loss / eval_iters
    model.train()
    return out


class Head(nn.Module):
    """A single attention head."""

    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))  # (B, T, C)
        x = x + self.ffwd(self.ln2(x))  # (B, T, C)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.positional_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            logits = logits.view(-1, logits.shape[-1])  # (B * T, C)
            targets = targets.view(-1)  # (B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the prediction for the next token
            logits, _ = self(idx_cond)
            # focus only on the last token
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)
            # sample the next token
            new_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append the new token to the sequence
            idx = torch.cat([idx, new_idx], dim=1)  # (B, T+1)
        return idx


model = GPTLanguageModel()
model = model.to(device)

# print the number of parameters in the model (M)
print(sum(p.numel() for p in model.parameters()) / 1e6)

# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        loss = estimate_loss()
        print(
            f"Iteration {iter}: train loss = {loss['train']:.4f}, val loss = {loss['val']:.4f}"
        )

    # get the data
    x, y = get_batch("train")

    # forward pass and compute the loss
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
decoded = decode(generated[0].tolist())
print(decoded)
