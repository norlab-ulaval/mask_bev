from torch.utils.data import Dataset


class ApplyTransform(Dataset):
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.dataset)
