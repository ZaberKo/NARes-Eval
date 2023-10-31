from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import pathlib
import PIL.Image

# According to "https://github.com/tml-epfl/understanding-fast-adv-training/blob/master/data.py", there is not problem with SVHN dataloader
# Datasets
available_datasets = ['CIFAR10', 'CIFAR100']

transform_options = {
    "CIFAR10": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(20),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
}


class DatasetGenerator():
    def __init__(self, train_batch_size=128, eval_batch_size=256, data_path='data/', dataset_type='CIFAR10',
                 num_of_workers=4, num_of_classes=10, input_size=32, use_augmentation=False, valset=False):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.input_size = input_size
        self.valset = valset  # Using CIFAR10.1 (1000 samples) as a separated validation set instead

        self.num_of_classes = num_of_classes
        if dataset_type == 'CIFAR10':
            name = 'cifar10'
            self.num_of_classes = 10
        elif dataset_type == 'CIFAR100':
            name = 'cifar100'
            self.num_of_classes = 100
        self.dataset_type = dataset_type
        self.data_path = "{}/{}".format(data_path, name)

        if self.dataset_type not in available_datasets:
            raise('Dataset type %s not implemented' % self.dataset_type)
        self.num_of_workers = num_of_workers
        self.use_augmentation = use_augmentation
        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        train_transform = transform_options[self.dataset_type]['train_transform']
        test_transform = transform_options[self.dataset_type]['test_transform']
        # change the input scale in transforms according to the input argument
        if 'CIFAR' in self.dataset_type:
            if self.input_size > 32:    # need to do some changes with the transforms
                train_transform.insert(0, transforms.Resize(self.input_size))
                train_transform[1] = transforms.RandomCrop(self.input_size, padding=4*(self.input_size//32))
                test_transform.insert(0, transforms.Resize(self.input_size))
        print(train_transform, test_transform)
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)

        if self.dataset_type == 'CIFAR10':
            self.num_of_classes = 10
            train_dataset = datasets.CIFAR10(root=self.data_path, train=True,
                                             transform=train_transform, download=True)
            # Adding CIFAR10.1 as validation support
            if self.valset:
                test_dataset = CIFAR101_test(self.data_path)
            else:
                test_dataset = datasets.CIFAR10(root=self.data_path, train=False,
                                            transform=test_transform, download=True)

        elif self.dataset_type == 'CIFAR100':
            self.num_of_classes = 100
            train_dataset = datasets.CIFAR100(root=self.data_path, train=True,
                                              transform=train_transform, download=True)

            test_dataset = datasets.CIFAR100(root=self.data_path, train=False,
                                             transform=test_transform, download=True)
        else:
            raise('Dataset type %s not implemented' % self.dataset_type)

        data_loaders = {
            'train_total': len(train_dataset),
            'test_total': len(test_dataset)
        }
        print(data_loaders)

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)
        data_loaders['train_set'], data_loaders['test_set'] = train_dataset, test_dataset
        data_loaders['train_batch_size'], data_loaders['test_batch_size'] = self.train_batch_size, self.eval_batch_size
        data_loaders['num_workers'] = self.num_of_workers
        return data_loaders


"""
This file implement the CIFAR-10.1. This dataset contains roughly 2,000 new test images that were sampled after 
multiple years of research on the original CIFAR-10 dataset. The data collection for CIFAR-10.1 was designed 
to minimize distribution shift relative to the original dataset.
"""


class CIFAR101_test(Dataset):
    def __init__(self, data_path, version_string='v6'):
        filename = 'cifar10.1'
        if version_string == '':
            version_string = 'v7'
        if version_string in ['v4', 'v6', 'v7']:
            filename += '_' + version_string
        else:
            raise ValueError('Unknown dataset version "{}".'.format(version_string))

        label_filename = filename + '_labels.npy'
        imagedata_filename = filename + '_data.npy'
        label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
        imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
        print('Loading labels from file {}'.format(label_filepath))
        assert pathlib.Path(label_filepath).is_file()
        labels = np.load(label_filepath)
        print('Loading image data from file {}'.format(imagedata_filepath))
        assert pathlib.Path(imagedata_filepath).is_file()
        imagedata = np.load(imagedata_filepath)
        self.len = imagedata.shape[0]
        assert len(labels.shape) == 1
        assert len(imagedata.shape) == 4
        assert labels.shape[0] == imagedata.shape[0]
        assert imagedata.shape[1] == 32
        assert imagedata.shape[2] == 32
        assert imagedata.shape[3] == 3
        if version_string == 'v6' or version_string == 'v7':
            assert labels.shape[0] == 2000
        elif version_string == 'v4':
            assert labels.shape[0] == 2021

        self.labels, self.imagedata = labels, imagedata
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img = self.imagedata[idx]
        img = np.uint8(img)
        img = PIL.Image.fromarray(img)
        label = self.labels[idx].astype(np.int64)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return self.len


