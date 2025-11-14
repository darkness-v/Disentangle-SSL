from train.py import train_epoch
from data_loader import ESDDataset
from nets import DisentangleSSLModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisentangleSSLModel(
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    dataset = ESDDataset(data_dir='data/Emotion Speech Dataset')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)

    for epoch in range(10):
        print(f"Epoch {epoch+1}")
        train_epoch(model, dataloader, optimizer, device)


if __name__ == "__main__":
    main()
