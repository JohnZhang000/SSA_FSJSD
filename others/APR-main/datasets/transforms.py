import torch
from torchvision import transforms

from datasets.APR import APRecombination

normalize = transforms.Compose([
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

def train_transforms(_transforms):
    transforms_list = []
    if 'aprs' in _transforms:
        print('APRecombination', _transforms)
        transforms_list.extend([
            transforms.RandomApply([APRecombination()], p=1.0),
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transforms_list.extend([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    return transforms_list


def test_transforms():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return test_transform

def train_transforms_tiny_imagenet(_transforms):
    transforms_list = []
    if 'aprs' in _transforms:
        print('APRecombination', _transforms)
        transforms_list.extend([
            transforms.RandomApply([APRecombination(img_size=64)], p=1.0),
            transforms.RandomCrop(64, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transforms_list.extend([
            transforms.RandomCrop(64, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    return transforms_list

def test_transforms_tiny_imagenet():
    test_transform = transforms.Compose([
        transforms.Resize(74),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])

    return test_transform