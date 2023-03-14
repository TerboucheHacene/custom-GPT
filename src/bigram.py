import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200

# manual seed
torch.manual_seed(1337)

# Read the data
with open(" input.txt", "r", encoding="utf-8") as f:
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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.emb(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            logits = logits.view(-1, logits.shape[-1])  # (B * T, C)
            targets = targets.view(-1)  # (B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            new_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, new_idx], dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
