import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import ESDDataset
from nets import DisentangleSSLModel
from tqdm import tqdm
import sys
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader):
        waveforms = [torch.tensor(sample).to(device) for sample in batch]
        optimizer.zero_grad()
        loss, _ = model(waveforms)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Average training loss: {avg_loss:.4f}")
    sys.stdout.flush()
    return avg_loss

