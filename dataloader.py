import os
import torch

from torchvision import datasets, transforms

def load_training(root_path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([40, 40]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485],
                              std=[0.229])
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(root_path), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

def load_testing(root_path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([40, 40]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485],
                              std=[0.229]),
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(root_path), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    return test_loader