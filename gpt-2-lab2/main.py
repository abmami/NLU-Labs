import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from datetime import timedelta


# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2


torch.manual_seed(1337)




with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of training or validation data for a language modeling task.

    Args:
    - split (str): The split to use for the data ('train' or 'val')

    Returns:
    - x (torch.Tensor): The input tensor with shape (batch_size, block_size)
    - y (torch.Tensor): The target tensor with shape (batch_size, block_size)
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Estimates the loss of a given model on the train and validation datasets.
    Returns:
    out : dict
        A dictionary containing the average loss for the train and validation datasets.
    """
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
    """
    One head of self-attention.
    
    Parameters:
        head_size (int): The size of the head.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # tril: lower triangle matrix or masking (it's not parameter and gets assigned the following way)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
    
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
            x (torch.Tensor): The input tensor of shape `(batch_size, seq_len, embed_dim)`.
        
        Returns:
            out (torch.Tensor): The output tensor of shape `(batch_size, seq_len, embed_dim)`.
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Weighted agregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T ,C)
        return out


class MultiHeadAtttention(nn.Module):
    """Multiple heads of self-attenton in parallel.

    Args:
        num_heads (int): The number of parallel self-attention heads.
        head_size (int): The size of each self-attention head.
    
    Attributes:
        heads (nn.ModuleList): The list of self-attention heads.
        proj (nn.Linear): The linear projection layer that transforms the concatenated
            self-attention head outputs to the original embedding size.
        dropout (nn.Dropout): The dropout layer to apply after the projection.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass for the multi-head self-attention.

        Args:
            x (torch.Tensor): The input tensor of shape `(batch_size, seq_len, n_embed)`.

        Returns:
            torch.Tensor: The output tensor of shape `(batch_size, seq_len, n_embed)`.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Applies a simple feedforward network to the input.

    Args:
        n_embed (int): The size of the input embedding.

    Attributes:
        net (nn.Sequential): The feedforward network consisting of two linear layers with ReLU activation
            and a final dropout layer.

    Methods:
        forward(x): Applies the feedforward network to the input tensor x.

    """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed)
        """
        return self.net(x)


class Block(nn.Module):
    """
    A Transformer block that performs self-attention and feedforward computation.

    Args:
    - n_embed (int): the size of the input vector
    - n_head (int): the number of attention heads to use

    Attributes:
    - sa (MultiHeadAtttention): a MultiHeadAtttention layer that performs self-attention
    - ffwd (FeedForward): a FeedForward layer that applies a non-linearity after a linear transformation
    - ln1 (nn.LayerNorm): a LayerNorm layer that normalizes the input after self-attention
    - ln2 (nn.LayerNorm): a LayerNorm layer that normalizes the input after the feedforward layer

    Methods:
    - forward(x): performs a forward pass through the block, taking in an input tensor x of shape (batch_size, seq_len, n_embed)
                  and returning an output tensor of the same shape after applying self-attention and feedforward computation.
    """

    def __init__(self, n_embed, n_head):
        super().__init__()
        # We cut the input vector of size n_embed into n_head chunks of head_size
        head_size = n_embed // n_head
        self.sa = MultiHeadAtttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 =nn.LayerNorm(n_embed)
        self.ln2 =nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    """
    A Transformer-based language model for predicting the next token in a sequence.

    Parameters:
        vocab_size (int): The size of the vocabulary.
        n_embed (int): The dimensionality of the token and position embeddings.
        block_size (int): The length of the input sequence.
        n_head (int): The number of attention heads in each Transformer block.
        n_layer (int): The number of Transformer blocks.
    
    Attributes:
        token_embedding_table (nn.Embedding): The embedding layer for the input tokens.
        position_embedding_table (nn.Embedding): The embedding layer for the position of each token.
        blocks (nn.Sequential): A sequence of Transformer blocks.
        ln_f (nn.LayerNorm): A layer normalization layer.
        lm_head (nn.Linear): A linear layer for predicting the next token.

    Methods:
        forward(idx, targets=None): Computes the forward pass of the model.
        generate(idx, max_new_tokens): Generates new tokens given an input sequence.
    """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)

    

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) # Predictions
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    


model = LanguageModel()
m = model.to(device)

print(f'{sum(p.numel() for p in m.parameters())} M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

start = time.time()

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses["train"]:.4f}, val los {losses["val"]:.4f}')
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end = time.time()
print('Training Time:', str(timedelta(seconds=end-start)))

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))