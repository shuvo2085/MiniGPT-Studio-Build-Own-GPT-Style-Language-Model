import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__(); assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1); hs = C // self.n_head
        q = q.view(B,T,self.n_head,hs).transpose(1,2)
        k = k.view(B,T,self.n_head,hs).transpose(1,2)
        v = v.view(B,T,self.n_head,hs).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) * (hs ** -0.5)
        scores = scores.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        y = torch.softmax(scores, dim=-1) @ v
        return self.proj(y.transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__(); self.ln1=nn.LayerNorm(n_embd); self.attn=CausalSelfAttention(n_embd,n_head,block_size)
        self.ln2=nn.LayerNorm(n_embd); self.ff=nn.Sequential(nn.Linear(n_embd,4*n_embd),nn.GELU(),nn.Linear(4*n_embd,n_embd))
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.ff(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embd=128, n_head=4, n_layer=4):
        super().__init__(); self.block_size=block_size
        self.token_emb=nn.Embedding(vocab_size,n_embd); self.pos_emb=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head,block_size) for _ in range(n_layer)])
        self.ln=nn.LayerNorm(n_embd); self.head=nn.Linear(n_embd,vocab_size)
    def forward(self, idx, targets=None):
        B,T=idx.shape; pos=torch.arange(T,device=idx.device)
        logits=self.head(self.ln(self.blocks(self.token_emb(idx)+self.pos_emb(pos))))
        loss=None
        if targets is not None: loss=nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
