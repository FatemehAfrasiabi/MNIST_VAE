import torch
import torchvision
from torchvision import transforms

class Util:
    def __init__(self) -> None:
        self.train_dataset = torchvision.datasets.MNIST('dataset', train=True, download=True)
        self.test_dataset = torchvision.datasets.MNIST('dataset', train=False, download=True)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomRotation([90, 180]),
            transforms.Resize([32, 32]),
            transforms.RandomCrop([28, 28]),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.train_dataset.transform = train_transform
        self.test_dataset.transform = test_transform

    def get_sep_indx_data(self, digit_filter, train=True):
        """Creates a dataset with only one specific class
        Args:
            digit_filter (int): class number
            train (bool): return train dataset
        Returns:
            MNIST dataset: MNIST dataset with only one class
        """
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
        sep_indices =  dataset.targets == digit_filter
        dataset.targets = dataset.targets[sep_indices]
        dataset.data = dataset.data[sep_indices]

        return dataset