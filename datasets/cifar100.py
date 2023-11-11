import numpy as np
from .sampler import ClassAwareSampler

import torch
import torchvision
from torchvision import transforms
import torchvision.datasets
import random
from PIL import Image,ImageFilter
from .autoaug import CIFAR10Policy, Cutout

class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, config, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.config = config
        self.gen_imbalanced_data(img_num_list)
        self.train = train
        self.dual_sample = config.sampler.dual_sample.enable
        if self.dual_sample:
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
        self.class_dict = self._get_class_dict()

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        meta = dict()
        if self.config.sampler.dual_sample.enable: #balance sampler
            assert self.config.sampler.weighted_sampler.type in ["balance", "reverse", "long-tailed"]
            if self.config.sampler.dual_sample.type == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.config.sampler.dual_sample.type == "balance":
                sample_class = random.randint(0, self.cls_num-1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.config.sampler.dual_sample.type == "long-tailed":
                sample_index = index #random.randint(0, self.__len__() - 1)
            
            sample_img, sample_label = self.data[sample_index], self.targets[sample_index]
            sample_img = Image.fromarray(sample_img)
            sample_img = self.transform(sample_img)

            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        if self.config.sampler.type == "weighted sampler" and self.train:
            assert self.config.sampler.weighted.type in ["balance", "reverse"]
            if  self.config.sampler.weighted.type == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.config.sampler.weighted.type == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        img, target = self.data[index], self.targets[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.dual_sample:
            return img, target, meta
        else:
            return img, target

    def get_num_classes(self):
        return self.cls_num

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos


    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


'''
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    """https://github.com/facebookresearch/moco"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
'''


class CIFAR100_LT(object):
    def __init__(self, config):
        '''
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
                
        norm_params = {'mean': [0.4914, 0.4822, 0.4465],
                       'std':  [0.2023, 0.1994, 0.2010]}
        '''        
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(config.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**norm_params),
                ])
             
        
        augmentation_sim_cifar = [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(**norm_params),
        ]        
        '''         
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),    # add AutoAug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Resize(config.image_size,antialias=False),
            transforms.Normalize(**norm_params),
            ])          
        
        
        val_transform = transforms.Compose([
                transforms.Resize((config.image_size * 8 // 7, config.image_size * 8 // 7),antialias=False),
                transforms.CenterCrop((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(**norm_params),
            ])        
        
        train_set = IMBALANCECIFAR100(config, root=config.data_path, imb_type=config.LT.imb_type, 
                                          imb_factor=config.LT.imb_factor, rand_number=0, 
                                          train=True, download=True, transform=train_transform)
        
        val_set = torchvision.datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=val_transform)
        
        self.train_len = len(train_set)
        self.val_len = len(val_set)

        
        self.cls_num_list = train_set.get_cls_num_list()
        self.num_classes = 100
        
        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if config.distributed else None
        self.train_loader = torch.utils.data.DataLoader(train_set,
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True, sampler=self.dist_sampler) ##drop_last=True

        balance_sampler = ClassAwareSampler(train_set)
        self.balance_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True, sampler=balance_sampler) 

        self.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True)
        

