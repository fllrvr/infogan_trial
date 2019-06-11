import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class LogoDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        image = np.asanyarray(image)
        dummy_y = np.zeros(1)

        sample = (image, dummy_y)

        return sample


def dataloader_(input_size, batch_size, dataset_name='mnist', data_root_dir='../'):
    transform = transforms.Compose(
                    [transforms.Resize((input_size, input_size)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    if dataset_name == 'mnist':
        dataloader = DataLoader(
            datasets.MNIST(os.path.join(data_root_dir, 'data/mnist'), 
                           train=True, download=True, transform=transform),
                           batch_size=batch_size, shuffle=True)

    elif dataset_name == 'logo':
        data_root_dir = '/home/ubuntu/dl_basic/'
        file_list = glob.glob(
            os.path.join(os.path.join(data_root_dir, 'data/logo_images/*')))
        logo_dataset = LogoDataset(file_list=file_list, transform=transform)
        dataloader = DataLoader(
            logo_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return dataloader
