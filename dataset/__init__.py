import torch
import pandas as pd
from torch.utils.data import Dataset

class FPDataset(Dataset):
    def __init__(self, csv_file: str, device = "cpu"):
        self.data = pd.read_csv(csv_file)
        self.xy = torch.tensor(self.data.iloc[:, :2].values, dtype=torch.int16).to(device)
        self.leds = torch.tensor(self.data.iloc[:, 2:].values, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        xy = self.xy[idx]
        leds = self.leds[idx]
        return leds, xy

__all__ = ["FPDataset"]