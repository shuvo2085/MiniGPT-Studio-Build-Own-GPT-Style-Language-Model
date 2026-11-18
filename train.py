from pathlib import Path
import torch
from model import MiniGPT
BASE=Path(__file__).resolve().parent
text=(BASE/'data'/'train.txt').read_text(encoding='utf-8')
chars=sorted(set(text))
if len(chars)<2: raise SystemExit('Training corpus needs at least two distinct characters.')
stoi={ch:i for i,ch in enumerate(chars)}
data=torch.tensor([stoi[c] for c in text],dtype=torch.long)
block_size=min(128, max(8, len(data)//4))
batch_size=16; steps=1000
device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
model=MiniGPT(len(chars),block_size=block_size).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
def get_batch():
    ix=torch.randint(0,len(data)-block_size-1,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y=torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x,y
if len(data)<=block_size+1: raise SystemExit('Upload a larger text corpus before training.')
model.train()
for step in range(steps):
    xb,yb=get_batch(); _,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    if step%100==0: print(f'step {step}/{steps} loss {loss.item():.4f}',flush=True)
torch.save({'model':model.state_dict(),'chars':chars,'config':{'vocab_size':len(chars),'block_size':block_size,'n_embd':128,'n_head':4,'n_layer':4}},BASE/'checkpoints'/'minigpt.pt')
print(f'Saved checkpoint. Device: {device}; vocabulary: {len(chars)}',flush=True)
print('Educational character-level model only; not comparable to ChatGPT.',flush=True)
