import torch
from torch.utils.data import Dataset

class VisitFeatureDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(row["feature_path"])

        x = data["fused_feature"].float()
        y = torch.tensor(int(row["label"]), dtype=torch.long)

        return x, y