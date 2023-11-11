import torch
import torchvision
from torchvision import transforms
from vtab.cifar import CIFAR10, CIFAR100
from vtab.flowers102 import Flowers102
from vtab.clevr import CLEVRClassification, CLEVRDistance
# --------Long Tailed Construction---------
from .cifar10 import CIFAR10_LT
from .cifar100 import CIFAR100_LT
from .places import Places_LT
from .imagenet import ImageNet_LT
from .ina2018 import iNa2018


class cifar100_1k_datasets(object):
    def __init__(self, config, download=True):

        
        norm_params = {'mean': [0.4914, 0.4822, 0.4465],
                       'std':  [0.2023, 0.1994, 0.2010]}
        normalize = transforms.Normalize(**norm_params)
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(config.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        val_transform = transforms.Compose([
                transforms.Resize((config.image_size * 8 // 7, config.image_size * 8 // 7)),
                transforms.CenterCrop((config.image_size, config.image_size)),
                transforms.ToTensor(),
                normalize,  
            ])
        
        
        train_set = CIFAR100(config.data_path, train=True, transform=train_transform, download=download, type='train1000')
        val_set = CIFAR100(config.data_path, train=False, transform=val_transform, download=download, type='test')
        self.num_classes = 100
        self.train_len = len(train_set)
        self.val_len = len(val_set)
        self.train_loader = torch.utils.data.DataLoader(
                    train_set,batch_size=config.batch_size, shuffle=True,
                    num_workers=config.workers, pin_memory=True) 

        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=config.batch_size, shuffle=False)        
        

def cifar100_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CIFAR100(data_path, train=True, transform=train_transforms, download=download, type='train800')
    val_set = CIFAR100(data_path, train=True, transform=val_transforms, download=download, type='val200')
    num_classes = 100
    return train_set, val_set, num_classes

def cifar100_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CIFAR100(data_path, train=True, transform=train_transforms, download=download, type='train1000')
    val_set = CIFAR100(data_path, train=False, transform=val_transforms, download=download, type='test')
    num_classes = 100
    return train_set, val_set, num_classes
    raise NotImplementedError





def flowers102_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 102
    return train_set, val_set, num_classes

def flowers102_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = Flowers102(data_path, split='val', transform=val_transforms, download=download, type='val200')
    num_classes = 102
    return train_set, val_set, num_classes

def flowers102_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='all')
    num_classes = 102
    return train_set, val_set, num_classes

#TODO: caltech-101 datasets
def caltech101_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    raise NotImplementedError
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 102
    return train_set, val_set, num_classes

def clevr_count_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRClassification(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = CLEVRClassification(data_path, split='val', transform=val_transforms, download=download, type='val')
    num_classes = 8
    return train_set, val_set, num_classes

def clevr_count_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRClassification(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = CLEVRClassification(data_path, split='train', transform=val_transforms, download=download, type='val200')
    num_classes = 8
    return train_set, val_set, num_classes

def clevr_count_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRClassification(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = CLEVRClassification(data_path, split='val', transform=val_transforms, download=download, type='all')
    num_classes = 8
    return train_set, val_set, num_classes

def clevr_distance_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRDistance(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = CLEVRDistance(data_path, split='val', transform=val_transforms, download=download, type='val')
    num_classes = 6
    return train_set, val_set, num_classes

def clevr_distance_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRDistance(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = CLEVRDistance(data_path, split='train', transform=val_transforms, download=download, type='val200')
    num_classes = 6
    return train_set, val_set, num_classes

def clevr_distance_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRDistance(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = CLEVRDistance(data_path, split='val', transform=val_transforms, download=download, type='all')
    num_classes = 6
    return train_set, val_set, num_classes

#TODO: Retinopathy datasets
def retinopathy_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    raise NotImplementedError
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 102
    return train_set, val_set, num_classes


def create_datasets(config): 
    if config.type == '1k':
        if config.dataset == 'cifar100': 
            dataset = cifar100_1k_datasets(config)
    elif config.type == 'LT':
        if config.dataset == 'cifar100':
            dataset = CIFAR100_LT(config)
        if config.dataset == 'imagenet':
            dataset = ImageNet_LT(config)
        if config.dataset == 'places':
            dataset = Places_LT(config)
    return dataset