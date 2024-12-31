"""
Unofficial code for VPT(Visual Prompt Tuning) paper of arxiv 2203.12119

A toy Tuning process that demostrates the code

the code is based on timm

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from PromptModels.GetPromptModel import build_promptmodel
import argparse


import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from timm.scheduler import CosineLRScheduler

from datasets.datasets import create_datasets
from utils import *
from loss import *
import time


def setup_seed(seed):  # setting up the random seed
    import random    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='VPT training')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='config/cifar100-LT-stage2.yaml',                        
                        #required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parse_args()
    setup_seed(42)
    save_path = os.path.join('./saved', config.name)
    #ensure_path(save_path)
    #set_log_path(save_path)
    model_dir, code_dir = create_path(save_path)  #LMK add, 0815
    save_code(code_dir)
    log('\n' + str(config))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.lr = config.base_lr * config.batch_size / 256 #config.base_lr * (config.batch_size / 256)**0.5 #

    dataset = create_datasets(config)
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
      
    log(f"train dataset: {dataset.train_len} samples")
    log(f"val dataset: {dataset.val_len} samples")
    config.num_classes = dataset.num_classes
    if config.loss.type == 'CrossEntropyLoss':
        config.normed = False
    else:
        config.normed = True  
    
    model = build_promptmodel(config, num_classes=config.num_classes, img_size=config.image_size, 
                              base_model=config.base_model, model_idx='ViT', patch_size=16,
                              Prompt_Token_num=config.prompt_length, VPT_type=config.VPT_type)  # VPT_type = "Shallow"
    
    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            log("=> loading checkpoint '{}'".format(config.resume))

            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            #log("=> loaded checkpoint '{}' (epoch {})"
            #            .format(config.resume, start_epoch))
        else:
            log("=> no checkpoint found at '{}'".format(config.resume))
    
    
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
    #scheduler = CosineLRScheduler(optimizer, warmup_lr_init=1e-5, t_initial=args.epochs, cycle_decay=0.1, warmup_t=args.warmup_epochs)
    scheduler = CosineLRScheduler(optimizer, t_initial=config.epochs, warmup_t=config.warmup_epochs, 
                                  warmup_lr_init=1e-4, decay_rate=config.decay_rate) #, cycle_decay=0.1  LMK:cycle_decay->delay_rate
    criterion = eval(config.loss.type)(config=config, cls_num_list=dataset.cls_num_list).to(device)

    # preds = model(data)  # (1, class_number)
    # print('before Tuning model output：', preds)

    # check backwarding tokens
    for param in model.parameters():
        if param.requires_grad:
            print(param.shape)
    max_va = -1
    best_epoch = -1
    for epoch in range(config.epochs):
        print('epoch:',epoch)
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: Averager() for k in aves_keys}

        aves = train(train_loader, model, criterion, optimizer, epoch, config, aves, device)
        aves = validate(val_loader, model, criterion, epoch, config, aves, device)
        print('After Tuning model output：', aves['va'].v)
        save_obj = {
            'config': vars(config),
            'state_dict': model.state_dict(),
            'val_acc': aves['va'].v,
        }
        if epoch <= config.epochs:
            torch.save(save_obj, os.path.join(model_dir, 'epoch-last.pth'))
            #torch.save(save_obj, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
    
            if aves['va'].v > max_va:
                max_va = aves['va'].v
                best_epoch = epoch
                torch.save(save_obj, os.path.join(model_dir, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(model_dir, 'epoch-ex.pth'))
        scheduler.step(epoch)
        #print('Best epoch', best_epoch)
        #print('Best Acc：', max_va)
        log_str = 'Best epoch {},  Best Acc {:.4f}'.format(best_epoch, max_va)
        log(log_str)   

def train(train_loader, model, criterion, optimizer, epoch, config, aves, device):
    model.Freeze_Prompt()
    #for imgs, targets in tqdm(train_loader, desc='train', leave=False):
    batch_time = Averager()  
    end = time.time()

    #training_data_num = len(train_loader.dataset)
    #end_steps = int(training_data_num / train_loader.batch_size)

    for iter_num, (imgs, targets) in enumerate(train_loader):       
        #if iter_num > end_steps:
        #    break
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        acc = compute_acc(outputs, targets)
        aves['tl'].add(loss.item())
        aves['ta'].add(acc)

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()        
        if iter_num  % config.print_freq == 0: 
            log_train_str = 'epoch {}, [{}/{}], train time {:.3f}|(loss) {:.4f}|(accuracy) {:.4f} '.format(
                    epoch, iter_num, len(train_loader), batch_time.v, aves['tl'].v, aves['ta'].v)
            log(log_train_str)     
            
    return  aves         
                

            
def validate(val_loader, model, criterion, epoch, config, aves, device):
    model.eval()  
    
    class_num = torch.zeros(config.num_classes).cuda()
    correct = torch.zeros(config.num_classes).cuda()
    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])
    with torch.no_grad():
        for iter_num, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.to(device)
            targets = targets.to(device).long()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            acc = compute_acc(outputs, targets)
            aves['vl'].add(loss.item())
            aves['va'].add(acc)
 
            _, predicted = outputs.max(1)
            target_one_hot = F.one_hot(targets, config.num_classes)            
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)            
            prob = torch.softmax(outputs, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, targets.cpu().numpy())

    acc_classes = correct / class_num
    head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() 
    med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() 
    tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() 
    cal = calibration(true_class, pred_class, confidence, num_bins=15)
    
    log_val_str = 'epoch {}, val (loss){:.4f}|(accuracy){:.4f}|(ECE){:.4f} \n'.format(
        epoch, aves['vl'].v, aves['va'].v, cal['expected_calibration_error'])
    
    log_val_str+= '\t\t HAcc {head_acc:.4f} MAcc {med_acc:.4f} TAcc {tail_acc:.4f}.'.format(
        head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc)
    log(log_val_str)
 
    return aves  
    


if __name__ == "__main__":
    main()
